"""Data caching layer to avoid re-scraping Basketball Reference on every run.

Scraped data is stored as Parquet files in data/raw/. Use --refresh or
set force_refresh=True to re-scrape and overwrite cached data.
"""

import os
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _ensure_cache_dir():
    """Create the cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(name, season):
    """Build the cache file path for a given data type and season.

    Args:
        name: Data type identifier (e.g., 'team_records', 'playoff_records').
        season: The NBA season year.

    Returns:
        Path to the Parquet cache file.
    """
    return CACHE_DIR / f"{name}_{season}.parquet"


def load_cached(name, season):
    """Load cached data for a given type and season, if available.

    Args:
        name: Data type identifier.
        season: The NBA season year.

    Returns:
        DataFrame if cache exists, None otherwise.
    """
    path = _cache_path(name, season)
    if path.exists():
        return pd.read_parquet(path)
    return None


def save_to_cache(df, name, season):
    """Save a DataFrame to the Parquet cache.

    Args:
        df: DataFrame to cache.
        name: Data type identifier.
        season: The NBA season year.
    """
    _ensure_cache_dir()
    path = _cache_path(name, season)
    df.to_parquet(path, index=False)


def is_cached(name, season):
    """Check if cached data exists for a given type and season.

    Args:
        name: Data type identifier.
        season: The NBA season year.

    Returns:
        bool: True if cache file exists.
    """
    return _cache_path(name, season).exists()


def clear_cache(name=None, season=None):
    """Clear cached data files.

    Args:
        name: If provided, only clear caches for this data type.
        season: If provided, only clear caches for this season.
            If both are None, clears all cached data.
    """
    _ensure_cache_dir()
    if name and season:
        path = _cache_path(name, season)
        if path.exists():
            path.unlink()
    elif name:
        for path in CACHE_DIR.glob(f"{name}_*.parquet"):
            path.unlink()
    elif season:
        for path in CACHE_DIR.glob(f"*_{season}.parquet"):
            path.unlink()
    else:
        for path in CACHE_DIR.glob("*.parquet"):
            path.unlink()
