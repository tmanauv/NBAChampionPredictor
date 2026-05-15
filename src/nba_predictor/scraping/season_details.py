from __future__ import annotations

from pathlib import Path

import pandas as pd

from nba_predictor.scraping.team_records import scrape_team_records
from nba_predictor.scraping.conf_standings import scrape_conf_standings
from nba_predictor.scraping.roster_accolades import scrape_roster_accolades
from nba_predictor.scraping.playoff_records import scrape_playoff_records


def scrape_season_details(
    season: int,
    cache_dir: Path | None = Path("data/cache"),
) -> pd.DataFrame:
    """Scrape and merge all data sources for a single *season*.

    Orchestrates calls to team_records, conf_standings, roster_accolades,
    and playoff_records, then joins them into a single DataFrame.
    """
    team_df, team_abrv = scrape_team_records(season, cache_dir=cache_dir)

    team_df = pd.merge(
        scrape_conf_standings(season, team_abrv, cache_dir=cache_dir),
        team_df,
        on="Team",
    )
    team_df.insert(1, "season", season)

    details_df = pd.merge(
        team_df,
        scrape_roster_accolades(season, team_abrv, cache_dir=cache_dir),
        on="Team",
    )
    details_df = pd.merge(
        details_df,
        scrape_playoff_records(season, team_abrv, cache_dir=cache_dir),
        on="Team",
    )

    for col in team_df.columns[1:]:
        if col not in ["Top3_Conf", "Conference"]:
            team_df[col] = team_df[col].replace(",", ".").astype(float)

    return details_df
