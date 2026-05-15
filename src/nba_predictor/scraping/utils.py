from __future__ import annotations

from typing import List, Tuple

import pandas as pd

TeamAbrv = List[Tuple[str, str]]


def map_team_names(
    df: pd.DataFrame,
    team_abrv: TeamAbrv,
    col: str = "Team",
) -> pd.DataFrame:
    """Replace full team names with three-letter abbreviations in *col*.

    Parameters
    ----------
    df : DataFrame
        Must contain *col* with full team names (e.g. ``"Milwaukee Bucks"``).
    team_abrv : list[tuple[str, str]]
        Pairs of ``(full_name, abbreviation)`` produced by
        :func:`scrape_team_records`.
    col : str
        Column name to modify.  Defaults to ``"Team"``.

    Returns
    -------
    DataFrame
        The same *df*, mutated in place for convenience.
    """
    name_to_abbr = {name: abbr for name, abbr in team_abrv}
    for idx in df.index:
        team = df.at[idx, col]
        if team in name_to_abbr:
            df.loc[idx, col] = name_to_abbr[team]
    return df


def fill_missing_teams(
    df: pd.DataFrame,
    team_abrv: TeamAbrv,
    fill_values: dict | None = None,
) -> pd.DataFrame:
    """Ensure every team in *team_abrv* has a row in *df*.

    Missing teams are appended with zeroed-out numeric columns (or custom
    *fill_values* keyed by column name).
    """
    abbreviations = {abbr for _, abbr in team_abrv}
    present = set(df["Team"].values)
    missing = abbreviations - present

    if not missing:
        return df

    cols = df.columns.tolist()
    rows = []
    for team in sorted(missing):
        row = {c: 0 for c in cols}
        row["Team"] = team
        if fill_values:
            row.update(fill_values)
        rows.append(row)

    return pd.concat([df, pd.DataFrame(rows, columns=cols)], ignore_index=True)
