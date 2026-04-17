"""Web scraping functions for NBA data from Basketball Reference."""

import re

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen


def get_team_abbreviations(soup_obj):
    """Extract team name-to-abbreviation mappings from a BeautifulSoup object.

    Returns:
        list[tuple[str, str]]: List of (team_name, abbreviation) tuples.
    """
    table_html = soup_obj.findAll("table", id=re.compile("advanced-team"))[0].findAll("a")

    team_name, team_abrv = [], []
    for html in table_html:
        abrv = html.get("href")
        pattern = re.compile(r"([A-Z]{3})")
        team_abrv.append(pattern.search(abrv).group())
        team_name.append(html.text)

    return list(zip(team_name, team_abrv))


def team_records(season):
    """Scrape advanced team records for a given NBA season.

    Args:
        season: The NBA season year (e.g., 2020 for the 2019-20 season).

    Returns:
        tuple: (team_records DataFrame, team_abbreviations list)
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}.html"
    soup_obj = BeautifulSoup(urlopen(url), "html.parser")

    team_abrv = get_team_abbreviations(soup_obj)

    table = soup_obj.findAll("table", id=re.compile("advanced-team"))
    records = pd.read_html(str(table))[0]

    records = records.apply(pd.to_numeric, errors="coerce").fillna(records)
    records.columns = records.columns.droplevel(0)
    records = records.drop(
        ["Rk", "PW", "PL", "Arena", "Attend.", "Attend./G"], axis=1
    )
    records["Team"] = records["Team"].str.replace("*", "", regex=False)

    for index in range(20, 24):
        if index != 22:
            records.columns.values[index] = "Opp_" + records.columns.values[index]

    records = records.drop(
        ["Unnamed: 17_level_1", "Unnamed: 22_level_1", "Unnamed: 27_level_1"], axis=1
    )
    records = records[:-1]

    team_names = [i[0] for i in team_abrv]
    team_abbreviations = [i[1] for i in team_abrv]

    for team, i in zip(records["Team"], records.index.values):
        if team in team_names:
            idx = team_names.index(team)
            records.loc[i, "Team"] = team_abbreviations[idx]

    for col in records.columns[1:]:
        records[col] = records[col].replace(",", ".").astype(float)

    records["SOY"] = (
        records.nlargest(5, "SRS")["SRS"].sum() / records["Team"].count()
    )

    records["Net_Four_Factors_Rating"] = (
        0.4 * records["eFG%"]
        - 0.25 * records["TOV%"]
        + 0.2 * records["ORB%"]
        + 0.15 * records["FT/FGA"]
    ) - (
        0.4 * records["Opp_eFG%"]
        - 0.25 * records["Opp_TOV%"]
        + 0.2 * (100 - records["DRB%"])
        + 0.15 * records["Opp_FT/FGA"]
    )
    records["Updated_Four_Factors_Rating"] = (
        0.5 * records["eFG%"]
        - 0.3 * records["TOV%"]
        + 0.15 * records["ORB%"]
        + 0.05 * records["FT/FGA"]
    ) - (
        0.5 * records["Opp_eFG%"]
        - 0.3 * records["Opp_TOV%"]
        + 0.15 * (100 - records["DRB%"])
        + 0.05 * records["Opp_FT/FGA"]
    )

    return records, team_abrv


def playoff_records(season, team_abrv):
    """Scrape playoff records for a given NBA season.

    Args:
        season: The NBA season year.
        team_abrv: List of (team_name, abbreviation) tuples.

    Returns:
        DataFrame with team playoff Champion_Share_Score.
    """
    url = f"https://www.basketball-reference.com/playoffs/NBA_{season}.html"

    table = BeautifulSoup(urlopen(url), "html.parser").findAll(
        "table", id=re.compile("advanced-team")
    )
    records = pd.read_html(str(table))[0]

    records = records.apply(pd.to_numeric, errors="coerce").fillna(records)
    records.columns = records.columns.droplevel(0)

    if "Team" in records.columns:
        records = records.loc[:, ["Team", "W", "L"]]
    else:
        records = records.loc[:, ["Tm", "W", "L"]]
        records.columns = ["Team", "W", "L"]

    records = records[:-1]
    records["Team"] = records["Team"].str.replace("*", "", regex=False)

    records["Champion_Share_Score"] = records["W"] / max(records["W"])

    drop_col = ["W", "L"]
    records.drop(drop_col, axis="columns", inplace=True)

    team_names = [i[0] for i in team_abrv]
    team_abbreviations = [i[1] for i in team_abrv]
    index_list = []

    for team, i in zip(records["Team"], records.index.values):
        if team in team_names:
            idx = team_names.index(team)
            index_list.append(idx)
            records["Team"] = records["Team"].replace(
                team, team_abbreviations[idx]
            )

    for team in team_abbreviations:
        if team_abbreviations.index(team) not in index_list:
            records.loc[len(records.index)] = [team, 0]

    records = records.sort_values(by=["Champion_Share_Score"], ascending=False)

    return records


def conf_standings(season, team_abrv):
    """Scrape conference standings for a given NBA season.

    Args:
        season: The NBA season year.
        team_abrv: List of (team_name, abbreviation) tuples.

    Returns:
        DataFrame with conference standings including Top3_Conf flag.
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_standings.html"

    table = BeautifulSoup(urlopen(url), "html.parser")
    econf_standings = pd.read_html(str(table))[0]
    wconf_standings = pd.read_html(str(table))[1]

    econf_standings.rename(columns={"Eastern Conference": "Team"}, inplace=True)
    wconf_standings.rename(columns={"Western Conference": "Team"}, inplace=True)

    econf_standings = econf_standings.loc[:, ["Team", "W/L%"]]
    wconf_standings = wconf_standings.loc[:, ["Team", "W/L%"]]

    econf_standings = econf_standings.apply(
        pd.to_numeric, errors="coerce"
    ).fillna(econf_standings)
    wconf_standings = wconf_standings.apply(
        pd.to_numeric, errors="coerce"
    ).fillna(wconf_standings)

    econf_standings["Team"] = econf_standings["Team"].str.replace(
        "*", "", regex=False
    )
    wconf_standings["Team"] = wconf_standings["Team"].str.replace(
        "*", "", regex=False
    )

    econf_standings["Top3_Conf"] = [
        True if i <= 2 else False for i in range(len(econf_standings))
    ]
    wconf_standings["Top3_Conf"] = [
        True if i <= 2 else False for i in range(len(wconf_standings))
    ]

    econf_standings["Conference"] = ["East" for _ in range(len(econf_standings))]
    wconf_standings["Conference"] = ["West" for _ in range(len(wconf_standings))]

    team_names = [i[0] for i in team_abrv]
    team_abbreviations = [i[1] for i in team_abrv]

    for team, i in zip(econf_standings["Team"], econf_standings.index.values):
        if team in team_names:
            idx = team_names.index(team)
            econf_standings.loc[i, "Team"] = team_abbreviations[idx]

    for team, i in zip(wconf_standings["Team"], wconf_standings.index.values):
        if team in team_names:
            idx = team_names.index(team)
            wconf_standings.loc[i, "Team"] = team_abbreviations[idx]

    standings = pd.concat([econf_standings, wconf_standings])

    return standings


def roster_accolades(season, team_abrv):
    """Scrape roster accolades (MVP, All-NBA, DPOY, All-Defense shares) for a season.

    Args:
        season: The NBA season year.
        team_abrv: List of (team_name, abbreviation) tuples.

    Returns:
        DataFrame with team-level accolade shares.
    """
    award_url = f"https://www.basketball-reference.com/awards/awards_{season}.html"
    soup_obj = BeautifulSoup(urlopen(award_url), "html.parser")

    mvp_table = soup_obj.findAll("table", id=re.compile("mvp"))
    mvp_details = pd.read_html(str(mvp_table))[0]
    mvp_details.columns = mvp_details.columns.droplevel(0)
    mvp_shares = mvp_details.loc[:, ["Tm", "Share"]]
    mvp_shares.rename(columns={"Tm": "Team", "Share": "mvp_share"}, inplace=True)
    mvp_shares = mvp_shares.groupby("Team", as_index=False).agg("sum")

    all_nba_table = soup_obj.findAll("table", id=re.compile("leading_all_nba"))
    all_nba_details = pd.read_html(str(all_nba_table))[0]
    all_nba_details.columns = all_nba_details.columns.droplevel(0)
    all_nba_shares = all_nba_details.loc[:, ["Tm", "Share"]]
    all_nba_shares.rename(
        columns={"Tm": "Team", "Share": "all_nba_share"}, inplace=True
    )
    all_nba_shares = all_nba_shares.groupby("Team", as_index=False).agg("sum")

    all_defense_table = soup_obj.findAll(
        "table", id=re.compile("leading_all_defense")
    )
    all_defense_details = pd.read_html(str(all_defense_table))[0]
    all_defense_details.columns = all_defense_details.columns.droplevel(0)
    all_defense_shares = all_defense_details.loc[:, ["Tm", "Share"]]
    all_defense_shares.rename(
        columns={"Tm": "Team", "Share": "all_defense_share"}, inplace=True
    )
    all_defense_shares = all_defense_shares.groupby("Team", as_index=False).agg("sum")

    dpoy_url = (
        f"https://www.basketball-reference.com/awards/awards_{season}.html#dpoy"
    )
    dpoy_table = BeautifulSoup(urlopen(dpoy_url), "html.parser")
    dpoy_details = pd.read_html(str(dpoy_table))[0]
    dpoy_details.columns = dpoy_details.columns.droplevel(0)
    dpoy_shares = dpoy_details.loc[:, ["Tm", "Share"]]
    dpoy_shares.rename(
        columns={"Tm": "Team", "Share": "dpoy_share"}, inplace=True
    )
    dpoy_shares = dpoy_shares.groupby("Team", as_index=False).agg("sum")

    accolades = pd.merge(mvp_shares, all_nba_shares, on="Team", how="outer")
    accolades = pd.merge(accolades, dpoy_shares, on="Team", how="outer")
    accolades = pd.merge(accolades, all_defense_shares, on="Team", how="outer")

    accolades = accolades.replace(np.nan, 0)

    team_abbreviations = [i[1] for i in team_abrv]

    index_list = []

    for team in accolades["Team"]:
        if team in team_abbreviations:
            idx = team_abbreviations.index(team)
            index_list.append(idx)

    for team in team_abbreviations:
        if team_abbreviations.index(team) not in index_list:
            accolades.loc[len(accolades.index)] = [team, 0.0, 0.0, 0.0, 0.0]

    return accolades


def playoff_experience(team, season):
    """Calculate cumulative playoff experience for a team's roster.

    Args:
        team: Team abbreviation (e.g., 'BOS').
        season: The NBA season year.

    Returns:
        int: Total playoff games played by current roster members before this season.
    """
    from time import sleep
    from tqdm.auto import tqdm

    roster_url = f"https://www.basketball-reference.com/teams/{team}/{season}.html"
    roster_table = BeautifulSoup(urlopen(roster_url), "html.parser")
    roster = pd.read_html(str(roster_table))[0]

    experience = 0

    for player in tqdm(roster["Player"], desc="roster loop"):
        name = player.split(" ")
        first_name = name[0].lower()
        second_name = name[1].lower()

        player_url = (
            f"https://www.basketball-reference.com/players/"
            f"{second_name[0]}/{second_name[:5]+first_name[:2]}01.html"
        )

        try:
            player_table = BeautifulSoup(urlopen(player_url), "html.parser").findAll(
                "table", id=re.compile("playoffs_totals")
            )
            sleep(1)
            player_details = pd.read_html(str(player_table))[0]

            player_details = player_details.loc[:, ["Season", "G", "MP"]]
            player_details = player_details[:-1]

            for sea in player_details["Season"]:
                start = sea[0:4]
                if int(start) >= season:
                    player_details.drop(
                        player_details[player_details["Season"] == sea].index,
                        inplace=True,
                    )

            experience += sum(player_details["G"])

        except (IndexError, ValueError, AttributeError):
            # Player page not found or no playoff data available
            pass

    return experience


def season_details(season):
    """Aggregate all data sources for a given season.

    Args:
        season: The NBA season year.

    Returns:
        DataFrame with complete season details for all teams.
    """
    records, team_abrv = team_records(season)

    team_df = pd.merge(conf_standings(season, team_abrv), records, on="Team")
    team_df.insert(1, "season", season)

    details = pd.merge(
        team_df, roster_accolades(season, team_abrv), on="Team"
    )

    # playoff_experience is commented out for performance reasons
    # Uncomment and optimize in a future PR
    # playoff_experience_dict = {}
    # for team in tqdm(details['Team'], desc='team loop'):
    #     playoff_exp = playoff_experience(team, season)
    #     sleep(5)
    #     playoff_experience_dict[team] = playoff_exp
    # details['Playoff_Experience'] = details['Team'].map(playoff_experience_dict)

    details = pd.merge(
        details, playoff_records(season, team_abrv), on="Team"
    )

    for col in team_df.columns[1:]:
        if col not in ["Top3_Conf", "Conference"]:
            team_df[col] = team_df[col].replace(",", ".").astype(float)

    return details
