from __future__ import annotations

from io import StringIO
from pathlib import Path
import re

from bs4 import BeautifulSoup
import pandas as pd

from nba_predictor.config import BASE_URL
from nba_predictor.scraping.http import fetch_html
from nba_predictor.scraping.utils import TeamAbrv, fill_missing_teams


def scrape_playoff_records(
    season: int,
    team_abrv: TeamAbrv,
    cache_dir: Path | None = Path("data/cache"),
) -> pd.DataFrame:
    """Scrape playoff win/loss records and compute Champion Share Score."""
    url = f"{BASE_URL}/playoffs/NBA_{season}.html"

    html = fetch_html(url, cache_dir=cache_dir)
    table = BeautifulSoup(html, "html.parser").findAll(
        "table", id=re.compile("advanced-team")
    )
    records_df = pd.read_html(StringIO(str(table)))[0]

    records_df = records_df.apply(pd.to_numeric, errors="coerce").fillna(records_df)
    records_df.columns = records_df.columns.droplevel(0)

    if "Team" in records_df.columns:
        records_df = records_df.loc[:, ["Team", "W", "L"]]
    else:
        records_df = records_df.loc[:, ["Tm", "W", "L"]]
        records_df.columns = ["Team", "W", "L"]

    records_df = records_df[:-1]
    records_df["Team"] = records_df["Team"].str.replace("*", "", regex=False)

    records_df["Champion_Share_Score"] = records_df["W"] / max(records_df["W"])
    records_df.drop(["W", "L"], axis="columns", inplace=True)

    name_to_abbr = {name: abbr for name, abbr in team_abrv}
    records_df["Team"] = records_df["Team"].replace(name_to_abbr)

    records_df = fill_missing_teams(records_df, team_abrv)
    records_df = records_df.sort_values(
        by=["Champion_Share_Score"], ascending=False
    )

    return records_df
