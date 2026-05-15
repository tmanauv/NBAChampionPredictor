from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from nba_predictor.config import BASE_URL
from nba_predictor.scraping.http import fetch_html
from nba_predictor.scraping.utils import TeamAbrv, map_team_names


def scrape_conf_standings(
    season: int,
    team_abrv: TeamAbrv,
    cache_dir: Path | None = Path("data/cache"),
) -> pd.DataFrame:
    """Scrape conference standings for *season*."""
    url = f"{BASE_URL}/leagues/NBA_{season}_standings.html"

    html = fetch_html(url, cache_dir=cache_dir)
    soup = BeautifulSoup(html, "html.parser")
    east_df = pd.read_html(StringIO(str(soup)))[0]
    west_df = pd.read_html(StringIO(str(soup)))[1]

    east_df.rename(columns={"Eastern Conference": "Team"}, inplace=True)
    west_df.rename(columns={"Western Conference": "Team"}, inplace=True)

    east_df = east_df.loc[:, ["Team", "W/L%"]]
    west_df = west_df.loc[:, ["Team", "W/L%"]]

    east_df = east_df.apply(pd.to_numeric, errors="coerce").fillna(east_df)
    west_df = west_df.apply(pd.to_numeric, errors="coerce").fillna(west_df)

    east_df["Team"] = east_df["Team"].str.replace("*", "", regex=False)
    west_df["Team"] = west_df["Team"].str.replace("*", "", regex=False)

    east_df["Top3_Conf"] = [i <= 2 for i in range(len(east_df))]
    west_df["Top3_Conf"] = [i <= 2 for i in range(len(west_df))]

    east_df["Conference"] = "East"
    west_df["Conference"] = "West"

    map_team_names(east_df, team_abrv)
    map_team_names(west_df, team_abrv)

    return pd.concat([east_df, west_df], ignore_index=True)
