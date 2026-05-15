from __future__ import annotations

from typing import List

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nba_predictor.config import CATEGORICAL_COLS


def build_preprocessor(
    feature_columns: List[str],
    categorical_cols: List[str] | None = None,
) -> ColumnTransformer:
    """Build a sklearn ``ColumnTransformer`` for the modelling pipeline.

    Categorical columns are one-hot encoded; the rest get median imputation
    followed by standard scaling.
    """
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS

    numeric_cols = [c for c in feature_columns if c not in categorical_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imp_median", SimpleImputer(missing_values=np.nan, strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(sparse_output=False), categorical_cols),
            ("num", numeric_pipe, numeric_cols),
        ]
    )


def get_header_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extract human-readable column names from a fitted preprocessor."""
    return [col.split("__")[1] for col in preprocessor.get_feature_names_out()]
