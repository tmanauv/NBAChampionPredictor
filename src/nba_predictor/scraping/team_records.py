from __future__ import annotations

import re
from io import StringIO
from pathlib import Path
from typing import Tuple

import pandas as pd
from bs4 import BeautifulSoup

from nba_predictor.config import BASE_URL
from nba_predictor.scraping.http import fetch_html
from nba_predictor.scraping.utils import TeamAbrv


def scrape_team_records(
    season: int,
    cache_dir: Path | None = Path("data/cache"),
) -> Tuple[pd.DataFrame, TeamAbrv]:
    """Scrape team advanced stats for *season* from Basketball Reference.

    Returns
    -------
    tuple[DataFrame, TeamAbrv]
        ``(records_df, team_abrv)`` where *team_abrv* is needed by the other
        scraping functions for the same season.
    """
    url = f"{BASE_URL}/leagues/NBA_{season}.html"
    html = fetch_html(url, cache_dir=cache_dir)
    soup = BeautifulSoup(html, "html.parser")
    links = soup.findAll("table", id=re.compile("advanced-team"))[0].findAll("a")

    team_name: list[str] = []
    team_abbr: list[str] = []
    for link in links:
        href = link.get("href")
        match = re.compile(r"([A-Z]{3})").search(href)
        team_abbr.append(match.group())
        team_name.append(link.text)

    team_abrv: TeamAbrv = list(zip(team_name, team_abbr))

    table = soup.findAll("table", id=re.compile("advanced-team"))
    records_df = pd.read_html(StringIO(str(table)))[0]

    records_df = records_df.apply(pd.to_numeric, errors="coerce").fillna(records_df)
    records_df.columns = records_df.columns.droplevel(0)
    records_df = records_df.drop(["Rk", "PW", "PL", "Arena", "Attend.", "Attend./G"], axis=1)
    records_df["Team"] = records_df["Team"].str.replace("*", "", regex=False)

    for index in range(20, 24):
        if index != 22:
            records_df.columns.values[index] = "Opp_" + records_df.columns.values[index]

    records_df = records_df.drop(
        ["Unnamed: 17_level_1", "Unnamed: 22_level_1", "Unnamed: 27_level_1"], axis=1
    )
    records_df = records_df[:-1]

    name_to_abbr = {name: abbr for name, abbr in team_abrv}
    for idx in records_df.index:
        team = records_df.at[idx, "Team"]
        if team in name_to_abbr:
            records_df.loc[idx, "Team"] = name_to_abbr[team]

    for col in records_df.columns[1:]:
        records_df[col] = records_df[col].replace(",", ".").astype(float)

    records_df["SOY"] = records_df.nlargest(5, "SRS")["SRS"].sum() / records_df["Team"].count()

    records_df["Net_Four_Factors_Rating"] = (
        0.4 * records_df["eFG%"]
        - 0.25 * records_df["TOV%"]
        + 0.2 * records_df["ORB%"]
        + 0.15 * records_df["FT/FGA"]
    ) - (
        0.4 * records_df["Opp_eFG%"]
        - 0.25 * records_df["Opp_TOV%"]
        + 0.2 * (100 - records_df["DRB%"])
        + 0.15 * records_df["Opp_FT/FGA"]
    )

    records_df["Updated_Four_Factors_Rating"] = (
        0.5 * records_df["eFG%"]
        - 0.3 * records_df["TOV%"]
        + 0.15 * records_df["ORB%"]
        + 0.05 * records_df["FT/FGA"]
    ) - (
        0.5 * records_df["Opp_eFG%"]
        - 0.3 * records_df["Opp_TOV%"]
        + 0.15 * (100 - records_df["DRB%"])
        + 0.05 * records_df["Opp_FT/FGA"]
    )

    return records_df, team_abrv
