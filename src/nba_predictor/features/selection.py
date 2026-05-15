from __future__ import annotations

from typing import List

import pandas as pd

from nba_predictor.config import TRAIN_CUTOFF


def select_features(
    seasons_df: pd.DataFrame,
    train_cutoff: int = TRAIN_CUTOFF,
) -> List[str]:
    """Return feature columns positively correlated with the target.

    Correlation is computed on the training window only
    (``season <= train_cutoff``) so holdout seasons do not influence
    which features are kept.

    Always includes ``Team``, ``season``, and ``Conference`` regardless of
    correlation.
    """
    train_df = seasons_df[seasons_df["season"] <= train_cutoff]
    corr = train_df.corrwith(
        train_df["Champion_Share_Score"], numeric_only=True
    )
    positively_correlated = list(corr[corr > 0].index)

    positively_correlated.insert(1, "Conference")
    positively_correlated.insert(0, "season")
    positively_correlated.insert(0, "Team")

    return positively_correlated
