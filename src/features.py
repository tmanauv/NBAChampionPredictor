"""Feature engineering and selection for NBA Champion prediction."""




def select_correlated_features(seasons_df):
    """Select features positively correlated with Champion_Share_Score.

    Args:
        seasons_df: DataFrame with all season data.

    Returns:
        list: Selected feature column names including metadata columns.
    """
    corr_series = seasons_df.corrwith(seasons_df["Champion_Share_Score"])
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
