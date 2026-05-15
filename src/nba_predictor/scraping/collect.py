from __future__ import annotations

from pathlib import Path
from time import sleep

import pandas as pd
from tqdm import tqdm

from nba_predictor.config import SEASON_START, SEASON_END, SCRAPE_DELAY_SECONDS
from nba_predictor.scraping.season_details import scrape_season_details


def collect_all_seasons(
    start: int = SEASON_START,
    end: int = SEASON_END,
    delay: float = SCRAPE_DELAY_SECONDS,
    cache_dir: Path | None = Path("data/cache"),
) -> pd.DataFrame:
    """Scrape season data from *end* down to *start* and return a combined DataFrame.

    Parameters
    ----------
    start : int
        First season (inclusive). Defaults to ``config.SEASON_START``.
    end : int
        Last season (inclusive). Defaults to ``config.SEASON_END``.
    delay : float
        Seconds to sleep between seasons to avoid rate-limiting.
    cache_dir : Path | None
        Directory for HTML cache.  ``None`` disables caching.
    """
    all_records: list[dict] = []

    for year in tqdm(range(end, start - 1, -1), desc="year loop"):
        season_df = scrape_season_details(year, cache_dir=cache_dir)
        sleep(delay)
        all_records += season_df.to_dict("records")

    return pd.DataFrame(all_records)
