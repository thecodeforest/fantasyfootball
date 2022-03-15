# def test_process_players.py
import pandas as pd
import pytest

from fantasyfootball.pipeline.process.process_players import process_players


@pytest.fixture(scope="module")
def df():
    columns = ["player", "team", "position", "season_year"]
    data = [
        ["Jonathan Taylor*+", "IND", "RB", 2021],
        ["Cooper Kupp*+", "LAR", "WR", 2021],
        ["Deebo Samuel*+", "SFO", "WR", 2021],
        ["Player", "Tm", "FantPos", 2021],
        ["Trenton Cannon", "2TM", "RB", 2021],
        ["John Wolford", "LAR", "QB", 2021],
    ]
    player_df = pd.DataFrame(data, columns=columns)
    return player_df


def test_process_players(df):
    columns = ["player", "tm", "position", "season_year"]
    expected = pd.DataFrame(
        [
            ["Jonathan Taylor", "IND", "RB", 2021],
            ["Cooper Kupp", "LAR", "WR", 2021],
            ["Deebo Samuel", "SFO", "WR", 2021],
            ["John Wolford", "LAR", "QB", 2021],
        ],
        columns=columns,
    )
    result = df.process_players()
    assert expected.equals(result)
