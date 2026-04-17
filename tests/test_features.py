"""Tests for the feature engineering module."""

import pandas as pd

from src.features import (
    add_engineered_features,
    build_correlation_matrix,
    select_correlated_features,
)


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


class TestAddEngineeredFeatures:
    def _make_df(self):
        return pd.DataFrame(
            {
                "Team": ["BOS", "LAL", "GSW"],
                "season": [2020, 2020, 2020],
                "W": [50, 45, 40],
                "L": [22, 27, 32],
                "SRS": [5.0, 2.0, -1.0],
                "MOV": [6.0, 2.5, -0.5],
                "ORtg": [112.0, 110.0, 108.0],
                "DRtg": [105.0, 108.0, 110.0],
                "Pace": [100.0, 98.0, 102.0],
                "FT%": [0.78, 0.75, 0.72],
                "3PAr": [0.35, 0.38, 0.40],
                "TS%": [0.58, 0.56, 0.54],
                "eFG%": [0.54, 0.52, 0.50],
                "TOV%": [12.0, 13.0, 14.0],
                "ORB%": [22.0, 20.0, 18.0],
                "FTr": [0.28, 0.25, 0.22],
            }
        )

    def test_win_pct(self):
        df = add_engineered_features(self._make_df())
        assert "Win_Pct" in df.columns
        assert abs(df.iloc[0]["Win_Pct"] - 50 / 72) < 0.01

    def test_net_rtg(self):
        df = add_engineered_features(self._make_df())
        assert "Net_Rtg" in df.columns
        assert df.iloc[0]["Net_Rtg"] == 7.0  # 112 - 105

    def test_srs_rank(self):
        df = add_engineered_features(self._make_df())
        assert "SRS_Rank" in df.columns
        assert df.iloc[0]["SRS_Rank"] == 1.0  # BOS has highest SRS

    def test_shooting_composite(self):
        df = add_engineered_features(self._make_df())
        assert "Shooting_Composite" in df.columns
        assert df["Shooting_Composite"].notna().all()

    def test_defensive_index(self):
        df = add_engineered_features(self._make_df())
        assert "Defensive_Index" in df.columns

    def test_mov_zscore(self):
        df = add_engineered_features(self._make_df())
        assert "MOV_Zscore" in df.columns

    def test_ftr_advantage(self):
        df = add_engineered_features(self._make_df())
        assert "FTr_Advantage" in df.columns
        # BOS has above-average FTr, so advantage should be positive
        assert df.iloc[0]["FTr_Advantage"] > 0

    def test_does_not_modify_original(self):
        original = self._make_df()
        original_cols = list(original.columns)
        add_engineered_features(original)
        assert list(original.columns) == original_cols

    def test_handles_missing_columns(self):
        df = pd.DataFrame({"Team": ["BOS"], "W": [50]})
        result = add_engineered_features(df)
        # Should not crash, just skip features that need missing columns
        assert "Team" in result.columns


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
