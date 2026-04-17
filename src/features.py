"""Feature engineering and selection for NBA Champion prediction."""

import numpy as np
import pandas as pd


def add_engineered_features(df):
    """Add derived features to the season DataFrame.

    Creates momentum, efficiency, and dominance features from existing columns.
    All new features are computed from columns already present in the scraped data.

    Args:
        df: DataFrame with season data (must include W, L, SRS, MOV, ORtg, DRtg,
            Pace, FT%, 3PAr, TS%, eFG%, TOV%, ORB%, FTr columns).

    Returns:
        DataFrame with additional engineered feature columns.
    """
    df = df.copy()

    # Win percentage (more granular than W alone)
    if "W" in df.columns and "L" in df.columns:
        total_games = df["W"] + df["L"]
        df["Win_Pct"] = np.where(total_games > 0, df["W"] / total_games, 0.0)

    # Net Rating = ORtg - DRtg (offensive minus defensive rating)
    if "ORtg" in df.columns and "DRtg" in df.columns:
        df["Net_Rtg"] = df["ORtg"] - df["DRtg"]

    # Pace-adjusted net rating (accounts for game tempo)
    if "Pace" in df.columns and "Net_Rtg" in df.columns:
        league_avg_pace = df["Pace"].mean()
        df["Pace_Adj_Net_Rtg"] = np.where(
            league_avg_pace > 0,
            df["Net_Rtg"] * (df["Pace"] / league_avg_pace),
            df["Net_Rtg"],
        )

    # SRS rank within the season (lower is better)
    if "SRS" in df.columns and "season" in df.columns:
        df["SRS_Rank"] = df.groupby("season")["SRS"].rank(
            ascending=False, method="min"
        )

    # Shooting efficiency composite: weighted blend of TS%, eFG%, FT%, 3PAr
    shooting_cols = ["TS%", "eFG%", "FT%", "3PAr"]
    if all(col in df.columns for col in shooting_cols):
        df["Shooting_Composite"] = (
            0.35 * df["TS%"]
            + 0.30 * df["eFG%"]
            + 0.20 * df["FT%"]
            + 0.15 * df["3PAr"]
        )

    # Defensive strength indicator: low DRtg + high TOV% forced + high ORB%
    if all(col in df.columns for col in ["DRtg", "TOV%", "ORB%"]):
        # Normalize DRtg (invert so lower = better becomes higher = better)
        df["Defensive_Index"] = (
            -0.50 * df["DRtg"] + 0.30 * df["TOV%"] + 0.20 * df["ORB%"]
        )

    # MOV (Margin of Victory) z-score within season
    if "MOV" in df.columns and "season" in df.columns:
        season_groups = df.groupby("season")["MOV"]
        df["MOV_Zscore"] = season_groups.transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
        )

    # Free throw rate advantage (FTr relative to league average)
    if "FTr" in df.columns and "season" in df.columns:
        season_avg_ftr = df.groupby("season")["FTr"].transform("mean")
        df["FTr_Advantage"] = df["FTr"] - season_avg_ftr

    return df


def select_correlated_features(seasons_df):
    """Select features positively correlated with Champion_Share_Score.

    Args:
        seasons_df: DataFrame with all season data.

    Returns:
        list: Selected feature column names including metadata columns.
    """
    numeric_df = seasons_df.select_dtypes(include="number")
    corr_series = numeric_df.corrwith(numeric_df["Champion_Share_Score"])
    selected_features = list(corr_series[corr_series > 0].index)
    selected_features.insert(1, "Conference")
    selected_features.insert(0, "season")
    selected_features.insert(0, "Team")
    return selected_features


def build_correlation_matrix(seasons_df):
    """Build a correlation matrix excluding non-numeric metadata columns.

    Args:
        seasons_df: DataFrame with all season data.

    Returns:
        DataFrame: Correlation matrix.
    """
    return seasons_df.drop(["Team", "season"], axis=1).corr()


def get_feature_splits(df):
    """Identify numeric and categorical feature columns.

    Args:
        df: Feature DataFrame (without target/metadata columns).

    Returns:
        tuple: (numeric_columns list, categorical_columns list)
    """
    categorical_cols = ["Top3_Conf", "Conference"]
    numeric_cols = [col for col in df.columns if col not in categorical_cols]
    return numeric_cols, categorical_cols


def select_features_mutual_info(X, y, k=15):
    """Select top-k features using mutual information regression.

    Mutual information captures non-linear relationships that Pearson
    correlation misses.

    Args:
        X: Feature DataFrame (numeric columns only).
        y: Target Series (Champion_Share_Score).
        k: Number of top features to select.

    Returns:
        list: Names of the top-k features ranked by mutual information.
    """
    from sklearn.feature_selection import mutual_info_regression

    X_filled = X.fillna(0)
    mi_scores = mutual_info_regression(X_filled, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    return list(mi_series.head(k).index)


def select_features_rfe(X, y, n_features=15):
    """Select features using Recursive Feature Elimination with GBR.

    Uses GradientBoostingRegressor as the estimator for RFE, which
    naturally handles feature importance ranking.

    Args:
        X: Feature DataFrame (numeric columns only).
        y: Target Series (Champion_Share_Score).
        n_features: Number of features to select.

    Returns:
        list: Names of selected features.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.feature_selection import RFE
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X), columns=X.columns, index=X.index
    )

    estimator = GradientBoostingRegressor(
        n_estimators=50, max_depth=3, random_state=42
    )
    selector = RFE(estimator, n_features_to_select=min(n_features, X.shape[1]))
    selector.fit(X_imputed, y)

    return list(X.columns[selector.support_])


def combined_feature_selection(seasons_df, target_col="Champion_Share_Score", k=15):
    """Select features using multiple methods and return their union.

    Combines correlation-based, mutual information, and RFE selection to
    get a robust feature set that captures both linear and non-linear
    relationships.

    Args:
        seasons_df: Full season DataFrame with all features.
        target_col: Name of the target column.
        k: Number of features to select per method.

    Returns:
        dict: {
            'correlation': list of correlation-selected features,
            'mutual_info': list of MI-selected features,
            'rfe': list of RFE-selected features,
            'union': combined unique feature list,
            'intersection': features selected by all methods,
        }
    """
    numeric_df = seasons_df.select_dtypes(include="number")
    feature_cols = [c for c in numeric_df.columns if c != target_col]
    X = numeric_df[feature_cols]
    y = numeric_df[target_col]

    # Method 1: Correlation
    corr_series = X.corrwith(y).dropna()
    corr_features = list(corr_series[corr_series > 0].nlargest(k).index)

    # Method 2: Mutual Information
    mi_features = select_features_mutual_info(X, y, k=k)

    # Method 3: RFE
    rfe_features = select_features_rfe(X, y, n_features=k)

    all_features = set(corr_features) | set(mi_features) | set(rfe_features)
    common_features = set(corr_features) & set(mi_features) & set(rfe_features)

    return {
        "correlation": corr_features,
        "mutual_info": mi_features,
        "rfe": rfe_features,
        "union": sorted(all_features),
        "intersection": sorted(common_features),
    }
