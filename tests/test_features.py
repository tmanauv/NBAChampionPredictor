"""Tests for the feature engineering module."""

import pandas as pd

from src.features import build_correlation_matrix, select_correlated_features


class TestSelectCorrelatedFeatures:
    def test_includes_metadata_columns(self):
        df = pd.DataFrame(
            {
                "Team": ["A", "B", "C", "D", "E"],
                "season": [2020, 2020, 2020, 2020, 2020],
                "W": [60, 50, 40, 30, 20],
                "L": [10, 20, 30, 40, 50],
                "Champion_Share_Score": [1.0, 0.8, 0.5, 0.2, 0.0],
            }
        )
        features = select_correlated_features(df)
        assert "Team" in features
        assert "season" in features
        assert "Conference" in features

    def test_excludes_negatively_correlated(self):
        df = pd.DataFrame(
            {
                "Team": ["A", "B", "C", "D", "E"],
                "season": [2020, 2020, 2020, 2020, 2020],
                "W": [60, 50, 40, 30, 20],
                "L": [10, 20, 30, 40, 50],
                "Champion_Share_Score": [1.0, 0.8, 0.5, 0.2, 0.0],
            }
        )
        features = select_correlated_features(df)
        # L is negatively correlated with Champion_Share_Score
        assert "L" not in features
        # W is positively correlated
        assert "W" in features


class TestBuildCorrelationMatrix:
    def test_excludes_metadata(self):
        df = pd.DataFrame(
            {
                "Team": ["BOS", "LAL"],
                "season": [2020, 2020],
                "W": [50, 45],
                "SRS": [3.5, 2.1],
            }
        )
        corr = build_correlation_matrix(df)
        assert "Team" not in corr.columns
        assert "season" not in corr.columns
        assert "W" in corr.columns
        assert "SRS" in corr.columns
