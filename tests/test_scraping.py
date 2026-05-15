from __future__ import annotations

from pathlib import Path

import pandas as pd

from nba_predictor.scraping.utils import TeamAbrv, fill_missing_teams, map_team_names

FIXTURES = Path(__file__).parent / "fixtures"


class TestMapTeamNames:
    def test_replaces_full_names_with_abbreviations(self):
        df = pd.DataFrame({"Team": ["Boston Celtics", "Denver Nuggets"]})
        team_abrv: TeamAbrv = [("Boston Celtics", "BOS"), ("Denver Nuggets", "DEN")]
        map_team_names(df, team_abrv)
        assert list(df["Team"]) == ["BOS", "DEN"]

    def test_leaves_unknown_names_unchanged(self):
        df = pd.DataFrame({"Team": ["Unknown Team"]})
        team_abrv: TeamAbrv = [("Boston Celtics", "BOS")]
        map_team_names(df, team_abrv)
        assert df["Team"].iloc[0] == "Unknown Team"

    def test_custom_column(self):
        df = pd.DataFrame({"Tm": ["Boston Celtics"]})
        team_abrv: TeamAbrv = [("Boston Celtics", "BOS")]
        map_team_names(df, team_abrv, col="Tm")
        assert df["Tm"].iloc[0] == "BOS"


class TestFillMissingTeams:
    def test_adds_missing_teams(self):
        df = pd.DataFrame({"Team": ["BOS"], "score": [1.0]})
        team_abrv: TeamAbrv = [("Boston Celtics", "BOS"), ("Denver Nuggets", "DEN")]
        result = fill_missing_teams(df, team_abrv)
        assert len(result) == 2
        assert set(result["Team"]) == {"BOS", "DEN"}

    def test_missing_teams_get_zero_values(self):
        df = pd.DataFrame({"Team": ["BOS"], "score": [1.0]})
        team_abrv: TeamAbrv = [("Boston Celtics", "BOS"), ("Denver Nuggets", "DEN")]
        result = fill_missing_teams(df, team_abrv)
        den_row = result[result["Team"] == "DEN"]
        assert den_row["score"].iloc[0] == 0

    def test_custom_fill_values(self):
        df = pd.DataFrame({"Team": ["BOS"], "score": [1.0]})
        team_abrv: TeamAbrv = [("Boston Celtics", "BOS"), ("Denver Nuggets", "DEN")]
        result = fill_missing_teams(df, team_abrv, fill_values={"score": -1.0})
        den_row = result[result["Team"] == "DEN"]
        assert den_row["score"].iloc[0] == -1.0

    def test_no_change_when_all_present(self):
        team_abrv: TeamAbrv = [("Boston Celtics", "BOS")]
        df = pd.DataFrame({"Team": ["BOS"], "score": [1.0]})
        result = fill_missing_teams(df, team_abrv)
        assert len(result) == 1


class TestConfStandingsFixture:
    """Test conf_standings parsing against fixture HTML."""

    def _fixture_fetcher(self, url: str, **kwargs) -> str:
        return (FIXTURES / "standings_NBA_2023.html").read_text()

    def test_parses_both_conferences(self, monkeypatch):
        monkeypatch.setattr(
            "nba_predictor.scraping.conf_standings.fetch_html", self._fixture_fetcher
        )
        from nba_predictor.scraping.conf_standings import scrape_conf_standings

        team_abrv: TeamAbrv = [
            ("Milwaukee Bucks", "MIL"),
            ("Boston Celtics", "BOS"),
            ("Denver Nuggets", "DEN"),
        ]
        df = scrape_conf_standings(2023, team_abrv)
        assert "Conference" in df.columns
        assert "Top3_Conf" in df.columns
        assert set(df["Conference"].unique()) == {"East", "West"}


class TestPlayoffRecordsFixture:
    """Test playoff_records parsing against fixture HTML."""

    def _fixture_fetcher(self, url: str, **kwargs) -> str:
        return (FIXTURES / "playoffs_NBA_2023.html").read_text()

    def test_champion_share_score_max_is_one(self, monkeypatch):
        monkeypatch.setattr(
            "nba_predictor.scraping.playoff_records.fetch_html", self._fixture_fetcher
        )
        from nba_predictor.scraping.playoff_records import scrape_playoff_records

        team_abrv: TeamAbrv = [
            ("Denver Nuggets", "DEN"),
            ("Boston Celtics", "BOS"),
            ("Milwaukee Bucks", "MIL"),
        ]
        df = scrape_playoff_records(2023, team_abrv)
        assert "Champion_Share_Score" in df.columns
        assert df["Champion_Share_Score"].max() == 1.0


class TestRosterAccoladesFixture:
    """Test roster_accolades parsing against fixture HTML."""

    def _fixture_fetcher(self, url: str, **kwargs) -> str:
        return (FIXTURES / "awards_2023.html").read_text()

    def test_returns_share_columns(self, monkeypatch):
        monkeypatch.setattr(
            "nba_predictor.scraping.roster_accolades.fetch_html", self._fixture_fetcher
        )
        from nba_predictor.scraping.roster_accolades import scrape_roster_accolades

        team_abrv: TeamAbrv = [
            ("Milwaukee Bucks", "MIL"),
            ("Boston Celtics", "BOS"),
            ("Denver Nuggets", "DEN"),
        ]
        df = scrape_roster_accolades(2023, team_abrv)
        assert "mvp_share" in df.columns
        assert "all_nba_share" in df.columns
        assert "dpoy_share" in df.columns
        assert "all_defense_share" in df.columns
