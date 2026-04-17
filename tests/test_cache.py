"""Tests for the data caching layer."""

import pandas as pd
import pytest

from src.cache import (
    clear_cache,
    is_cached,
    load_cached,
    save_to_cache,
)


@pytest.fixture(autouse=True)
def clean_cache(tmp_path, monkeypatch):
    """Use a temporary directory for cache during tests."""
    monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
    yield
    # Cleanup happens automatically with tmp_path


class TestSaveAndLoad:
    def test_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        df = pd.DataFrame({"Team": ["BOS", "LAL"], "W": [50, 45]})
        save_to_cache(df, "team_records", 2020)
        loaded = load_cached("team_records", 2020)
        assert loaded is not None
        pd.testing.assert_frame_equal(df, loaded)

    def test_load_missing_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        assert load_cached("nonexistent", 9999) is None

    def test_is_cached(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        assert not is_cached("team_records", 2020)
        df = pd.DataFrame({"Team": ["BOS"]})
        save_to_cache(df, "team_records", 2020)
        assert is_cached("team_records", 2020)


class TestClearCache:
    def test_clear_specific(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        df = pd.DataFrame({"Team": ["BOS"]})
        save_to_cache(df, "team_records", 2020)
        save_to_cache(df, "team_records", 2021)
        clear_cache("team_records", 2020)
        assert not is_cached("team_records", 2020)
        assert is_cached("team_records", 2021)

    def test_clear_by_name(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        df = pd.DataFrame({"Team": ["BOS"]})
        save_to_cache(df, "team_records", 2020)
        save_to_cache(df, "team_records", 2021)
        save_to_cache(df, "playoff_records", 2020)
        clear_cache(name="team_records")
        assert not is_cached("team_records", 2020)
        assert not is_cached("team_records", 2021)
        assert is_cached("playoff_records", 2020)

    def test_clear_by_season(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        df = pd.DataFrame({"Team": ["BOS"]})
        save_to_cache(df, "team_records", 2020)
        save_to_cache(df, "playoff_records", 2020)
        save_to_cache(df, "team_records", 2021)
        clear_cache(season=2020)
        assert not is_cached("team_records", 2020)
        assert not is_cached("playoff_records", 2020)
        assert is_cached("team_records", 2021)

    def test_clear_all(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        df = pd.DataFrame({"Team": ["BOS"]})
        save_to_cache(df, "team_records", 2020)
        save_to_cache(df, "playoff_records", 2021)
        clear_cache()
        assert not is_cached("team_records", 2020)
        assert not is_cached("playoff_records", 2021)
