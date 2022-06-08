import pytest
import pandas as pd
import numpy as np
from fantasyfootball.config import root_dir, data_sources, scoring
from fantasyfootball.data import FantasyData
from urllib.error import HTTPError


@pytest.fixture(scope="module")
def df():
    columns = [
        "name",
        "pid",
        "date",
        "position",
        "passing_yds",
        "passing_td",
        "passing_int",
        "rushing_yds",
    ]
    data = [
        ["John Smith", "Jsmit01", "2019-01-01", "QB", 100, 1, 0, 100],
        ["John Smith", "Jsmit01", "2019-08-01", "QB", 100, 2, 1, 0],
    ]
    player_df = pd.DataFrame(data, columns=columns)
    return player_df


def test__validate_season_year_range():
    season_year_start = 2017
    season_year_end = 2020
    expected = True
    fantasy_data = FantasyData(season_year_start, season_year_end)
    result = fantasy_data._validate_season_year_range()
    assert result == expected


def test__validate_season_year_range_error():
    season_year_start = 2000  # min season start is 2015
    season_year_end = 2020
    with pytest.raises(ValueError):
        FantasyData(season_year_start, season_year_end)


def test__refresh_data():
    season_year = 2020
    ff_data_dir = root_dir / "datasets" / "season" / str(season_year)
    expected = True
    _refresh_data = FantasyData._refresh_data
    result = _refresh_data(ff_data_dir, data_sources)
    assert result == expected


# TO DO:Once the data is refreshed, test that the data is refreshed.
# def test__refresh_data_error():
#     season_year = 2999
#     ff_data_dir = root_dir / "datasets" / "season" / str(season_year)
#     _refresh_data = FantasyData._refresh_data
#     with pytest.raises(HTTPError):
#         _refresh_data(ff_data_dir, data_sources)


def test_add_scoring_source():
    fantasy_data = FantasyData(season_year_start=2019, season_year_end=2021)
    scoring_source = "test_league"
    new_scoring_source = {
        scoring_source: {
            "scoring_columns": {
                "passing_td": 4,
                "passing_yds": 0.04,
                "passing_int": -3,
                "rushing_td": 6,
                "rushing_yds": 0.1,
                "receiving_rec": 0.5,
                "receiving_td": 4,
                "receiving_yds": 0.1,
                "fumbles_fmb": -3,
                "scoring_2pm": 4,
                "punt_returns_td": 6,
            },
            "multiplier": {
                "rushing_yds": {"threshold": 100, "points": 5},
                "passing_yds": {"threshold": 300, "points": 3},
                "receiving_yds": {"threshold": 100, "points": 3},
            },
        }
    }
    fantasy_data.add_scoring_source(new_scoring_source)
    assert scoring_source in fantasy_data.scoring.keys()


def test_score_player(df):
    scoring_source = "yahoo"
    scoring_columns = {"passing_yds", "passing_td", "passing_int", "rushing_yds"}
    scoring_source_rules = scoring.get(scoring_source)
    score_player = FantasyData.score_player
    expected = [18, 11]
    result = list(score_player(df, scoring_columns, scoring_source_rules))
    assert result == expected


def test_create_fantasy_points_column(df):
    scoring_source = "yahoo"
    season_year_start = 2020
    season_year_end = 2020
    fantasy_data = FantasyData(season_year_start, season_year_end)
    # replace default data with test data
    fantasy_data.ff_data = df
    fantasy_data.create_fantasy_points_column(scoring_source)
    expected = [18, 11]
    result = fantasy_data.ff_data[fantasy_data.ff_data.columns[-1]].tolist()
    assert result == expected
