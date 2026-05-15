from __future__ import annotations

from io import StringIO
from urllib.request import urlopen

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import re

from nba_predictor.config import BASE_URL
from nba_predictor.scraping.utils import TeamAbrv, fill_missing_teams


def scrape_roster_accolades(season: int, team_abrv: TeamAbrv) -> pd.DataFrame:
    """Scrape MVP, All-NBA, DPOY, and All-Defense shares for *season*."""
    award_url = f"{BASE_URL}/awards/awards_{season}.html"
    soup = BeautifulSoup(urlopen(award_url), "html.parser")

    mvp_table = soup.findAll("table", id=re.compile("mvp"))
    mvp_details = pd.read_html(StringIO(str(mvp_table)))[0]
    mvp_details.columns = mvp_details.columns.droplevel(0)
    mvp_shares = mvp_details.loc[:, ["Tm", "Share"]]
    mvp_shares.rename(columns={"Tm": "Team", "Share": "mvp_share"}, inplace=True)
    mvp_shares = mvp_shares.groupby("Team", as_index=False).agg("sum")

    all_nba_table = soup.findAll("table", id=re.compile("leading_all_nba"))
    all_nba_details = pd.read_html(StringIO(str(all_nba_table)))[0]
    all_nba_details.columns = all_nba_details.columns.droplevel(0)
    all_nba_shares = all_nba_details.loc[:, ["Tm", "Share"]]
    all_nba_shares.rename(
        columns={"Tm": "Team", "Share": "all_nba_share"}, inplace=True
    )
    all_nba_shares = all_nba_shares.groupby("Team", as_index=False).agg("sum")

    all_defense_table = soup.findAll("table", id=re.compile("leading_all_defense"))
    all_defense_details = pd.read_html(StringIO(str(all_defense_table)))[0]
    all_defense_details.columns = all_defense_details.columns.droplevel(0)
    all_defense_shares = all_defense_details.loc[:, ["Tm", "Share"]]
    all_defense_shares.rename(
        columns={"Tm": "Team", "Share": "all_defense_share"}, inplace=True
    )
    all_defense_shares = all_defense_shares.groupby("Team", as_index=False).agg("sum")

    dpoy_url = f"{BASE_URL}/awards/awards_{season}.html#dpoy"
    dpoy_soup = BeautifulSoup(urlopen(dpoy_url), "html.parser")
    dpoy_details = pd.read_html(StringIO(str(dpoy_soup)))[0]
    dpoy_details.columns = dpoy_details.columns.droplevel(0)
    dpoy_shares = dpoy_details.loc[:, ["Tm", "Share"]]
    dpoy_shares.rename(columns={"Tm": "Team", "Share": "dpoy_share"}, inplace=True)
    dpoy_shares = dpoy_shares.groupby("Team", as_index=False).agg("sum")

    accolades_df = pd.merge(mvp_shares, all_nba_shares, on="Team", how="outer")
    accolades_df = pd.merge(accolades_df, dpoy_shares, on="Team", how="outer")
    accolades_df = pd.merge(accolades_df, all_defense_shares, on="Team", how="outer")
    accolades_df = accolades_df.replace(np.nan, 0)

    accolades_df = fill_missing_teams(
        accolades_df,
        team_abrv,
        fill_values={
            "mvp_share": 0.0,
            "all_nba_share": 0.0,
            "dpoy_share": 0.0,
            "all_defense_share": 0.0,
        },
    )

    return accolades_df
