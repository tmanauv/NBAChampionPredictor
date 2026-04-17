"""Tests for the HTTP client module."""

import pandas as pd
import pytest

from src.http_client import validate_dataframe


class TestValidateDataframe:
    def test_valid_dataframe(self):
        df = pd.DataFrame({"Team": ["BOS", "LAL"], "W": [50, 45]})
        # Should not raise
        validate_dataframe(df, expected_columns=["Team", "W"], min_rows=2, name="test")

    def test_empty_dataframe_raises(self):
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame is empty"):
            validate_dataframe(df, name="test")

    def test_none_dataframe_raises(self):
        with pytest.raises(ValueError, match="DataFrame is empty"):
            validate_dataframe(None, name="test")

    def test_too_few_rows_raises(self):
        df = pd.DataFrame({"Team": ["BOS"]})
        with pytest.raises(ValueError, match="Expected at least 5 rows"):
            validate_dataframe(df, min_rows=5, name="test")

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"Team": ["BOS"], "W": [50]})
        with pytest.raises(ValueError, match="Missing expected columns"):
            validate_dataframe(
                df, expected_columns=["Team", "W", "L"], name="test"
            )

    def test_high_nan_warning(self, caplog):
        import logging

        df = pd.DataFrame({"Team": ["BOS", "LAL", "GSW"], "W": [None, None, 50]})
        with caplog.at_level(logging.WARNING):
            validate_dataframe(df, name="test")
        assert "High NaN ratio" in caplog.text


class TestFeatureSplits:
    def test_get_feature_splits(self):
        from src.features import get_feature_splits

        df = pd.DataFrame(
            {
                "Top3_Conf": [True, False],
                "Conference": ["East", "West"],
                "W": [50, 45],
                "SRS": [3.5, 2.1],
            }
        )
        numeric, categorical = get_feature_splits(df)
        assert "Top3_Conf" in categorical
        assert "Conference" in categorical
        assert "W" in numeric
        assert "SRS" in numeric
