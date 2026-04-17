#!/usr/bin/env python3
"""CLI script for NBA Champion prediction.

Usage:
    python predict.py --season 2024
    python predict.py --season 2024 --retrain
    python predict.py --season 2024 --refresh
    python predict.py --season 2024 --top 10
"""

import argparse
import logging
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.cache import clear_cache
from src.evaluation import save_model, load_model
from src.features import add_engineered_features, select_correlated_features
from src.model import build_preprocessor
from src.scraping import season_details
from src.tuning import tune_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "champion_predictor"
TRAIN_START_SEASON = 1980
TRAIN_END_SEASON = 2020


def scrape_seasons(start, end, force_refresh=False):
    """Scrape season data for a range of years.

    Args:
        start: First season year.
        end: Last season year (inclusive).
        force_refresh: If True, bypass cache.

    Returns:
        DataFrame with all seasons combined.
    """
    all_seasons = []
    for season in tqdm(range(start, end + 1), desc="Scraping seasons"):
        try:
            details = season_details(season, force_refresh=force_refresh)
            all_seasons.append(details)
        except Exception as e:
            logger.warning("Failed to scrape season %d: %s", season, e)
            continue

    if not all_seasons:
        logger.error("No seasons scraped successfully")
        sys.exit(1)

    return pd.concat(all_seasons, ignore_index=True)


def train_model(seasons_df):
    """Train the best model using Optuna tuning.

    Args:
        seasons_df: DataFrame with all season data.

    Returns:
        tuple: (fitted_model, preprocessor, feature_columns)
    """
    logger.info("Engineering features...")
    seasons_df = add_engineered_features(seasons_df)

    logger.info("Selecting features...")
    selected = select_correlated_features(seasons_df)
    seasons_df = seasons_df[[c for c in selected if c in seasons_df.columns]]

    X = seasons_df.drop(["Team", "season", "Champion_Share_Score"], axis=1)
    y = seasons_df["Champion_Share_Score"]

    logger.info("Building preprocessor...")
    preprocessor = build_preprocessor(X)
    X_transformed = preprocessor.fit_transform(X)

    logger.info("Tuning GradientBoosting with Optuna (50 trials)...")
    result = tune_model("GradientBoosting", X_transformed, y, n_trials=50)
    model = result["best_model"]

    logger.info(
        "Best model: MAE=%.4f, params=%s",
        result["best_score"],
        result["best_params"],
    )

    feature_cols = list(X.columns)

    # Save the model
    metadata = {
        "best_params": result["best_params"],
        "best_score": result["best_score"],
        "feature_columns": feature_cols,
        "train_seasons": f"{seasons_df['season'].min()}-{seasons_df['season'].max()}"
        if "season" in seasons_df.columns
        else "unknown",
        "trained_at": datetime.now().isoformat(),
    }
    save_model(model, DEFAULT_MODEL_NAME, metadata)

    return model, preprocessor, feature_cols


def predict_season(model, preprocessor, feature_cols, season, force_refresh=False):
    """Generate predictions for a specific season.

    Args:
        model: Fitted model.
        preprocessor: Fitted ColumnTransformer.
        feature_cols: List of feature column names.
        season: Season year to predict.
        force_refresh: If True, bypass cache for scraping.

    Returns:
        DataFrame with team predictions sorted by predicted championship share.
    """
    logger.info("Scraping data for season %d...", season)
    season_df = season_details(season, force_refresh=force_refresh)
    season_df = add_engineered_features(season_df)

    teams = season_df["Team"].values

    # Select only the features the model was trained on
    available_cols = [c for c in feature_cols if c in season_df.columns]
    missing_cols = [c for c in feature_cols if c not in season_df.columns]
    if missing_cols:
        logger.warning("Missing features (will be NaN): %s", missing_cols)

    X = season_df[available_cols].copy()
    for col in missing_cols:
        X[col] = np.nan

    X = X[feature_cols]  # Ensure correct column order
    X_transformed = preprocessor.transform(X)

    predictions = model.predict(X_transformed)

    results_df = pd.DataFrame(
        {
            "Team": teams,
            "Championship_Share_%": np.round(predictions * 100, 2),
        }
    )
    results_df = results_df.sort_values(
        "Championship_Share_%", ascending=False
    ).reset_index(drop=True)
    results_df.index += 1  # 1-based ranking
    results_df.index.name = "Rank"

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="NBA Champion Predictor — predict championship contenders"
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season year to predict (e.g., 2024)",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retrain the model (even if a saved model exists)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-scrape data (bypass cache)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top teams to display (default: 5)",
    )
    parser.add_argument(
        "--train-start",
        type=int,
        default=TRAIN_START_SEASON,
        help=f"First training season (default: {TRAIN_START_SEASON})",
    )
    parser.add_argument(
        "--train-end",
        type=int,
        default=TRAIN_END_SEASON,
        help=f"Last training season (default: {TRAIN_END_SEASON})",
    )

    args = parser.parse_args()

    if args.refresh:
        logger.info("Clearing cache...")
        clear_cache()

    # Try to load existing model, or train a new one
    model = None
    preprocessor = None
    feature_cols = None

    if not args.retrain:
        try:
            model, metadata = load_model(DEFAULT_MODEL_NAME)
            feature_cols = metadata.get("feature_columns") if metadata else None
            if feature_cols:
                logger.info(
                    "Loaded saved model (trained on %s)",
                    metadata.get("train_seasons", "unknown"),
                )
            else:
                logger.warning("Saved model has no feature metadata, retraining...")
                model = None
        except FileNotFoundError:
            logger.info("No saved model found, training...")

    if model is None or feature_cols is None:
        logger.info(
            "Training on seasons %d-%d...", args.train_start, args.train_end
        )
        seasons_df = scrape_seasons(
            args.train_start, args.train_end, force_refresh=args.refresh
        )
        model, preprocessor, feature_cols = train_model(seasons_df)

    if preprocessor is None:
        # Rebuild preprocessor from training data
        logger.info("Rebuilding preprocessor...")
        seasons_df = scrape_seasons(args.train_start, args.train_end)
        seasons_df = add_engineered_features(seasons_df)
        selected = select_correlated_features(seasons_df)
        seasons_df = seasons_df[[c for c in selected if c in seasons_df.columns]]
        X = seasons_df.drop(["Team", "season", "Champion_Share_Score"], axis=1)
        preprocessor = build_preprocessor(X)
        preprocessor.fit(X)

    # Predict
    results = predict_season(
        model, preprocessor, feature_cols, args.season, force_refresh=args.refresh
    )

    print(f"\n{'='*50}")
    print(f"  NBA Champion Predictions — {args.season} Season")
    print(f"{'='*50}\n")
    print(results.head(args.top).to_string())
    print(f"\n{'='*50}")
    print(f"Showing top {args.top} of {len(results)} teams")
    print()


if __name__ == "__main__":
    main()
