from __future__ import annotations

import numpy as np
import pandas as pd

from nba_predictor.features.preprocessing import build_preprocessor, get_header_names
from nba_predictor.features.selection import select_features


class TestSelectFeatures:
    def test_includes_required_columns(self):
        df = pd.DataFrame(
            {
                "Team": ["A", "B", "C", "D"],
                "season": [2018, 2019, 2020, 2021],
                "Conference": ["East", "West", "East", "West"],
                "W": [50, 60, 55, 45],
                "L": [32, 22, 27, 37],
                "Champion_Share_Score": [0.2, 0.8, 0.6, 0.1],
            }
        )
        features = select_features(df, train_cutoff=2020)
        assert "Team" in features
        assert "season" in features
        assert "Conference" in features

    def test_uses_only_train_data(self):
        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame(
            {
                "Team": [f"T{i}" for i in range(n)],
                "season": [2015 + (i % 10) for i in range(n)],
                "Conference": ["East"] * n,
                "good_feature": rng.uniform(0, 1, n),
                "noise": rng.uniform(-1, 1, n),
                "Champion_Share_Score": rng.uniform(0, 1, n),
            }
        )
        df["good_feature"] = df["Champion_Share_Score"] + rng.normal(0, 0.1, n)
        features = select_features(df, train_cutoff=2020)
        assert "good_feature" in features


class TestPreprocessor:
    def test_build_and_transform(self):
        feature_cols = ["Top3_Conf", "Conference", "W", "SRS"]
        preprocessor = build_preprocessor(feature_cols)

        df = pd.DataFrame(
            {
                "Top3_Conf": [True, False, True],
                "Conference": ["East", "West", "East"],
                "W": [50.0, 60.0, 55.0],
                "SRS": [3.0, 5.0, 4.0],
            }
        )
        result = preprocessor.fit_transform(df)
        assert result.shape[0] == 3
        assert result.shape[1] > 4  # one-hot expands categoricals

    def test_get_header_names(self):
        feature_cols = ["Top3_Conf", "Conference", "W"]
        preprocessor = build_preprocessor(feature_cols)

        df = pd.DataFrame(
            {
                "Top3_Conf": [True, False],
                "Conference": ["East", "West"],
                "W": [50.0, 60.0],
            }
        )
        preprocessor.fit(df)
        names = get_header_names(preprocessor)
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)
        assert len(names) > 0
