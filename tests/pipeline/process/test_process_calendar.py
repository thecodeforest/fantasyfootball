from pathlib import Path
import pandas as pd
import pytest
import sys

sys.path.append(str(Path.cwd()))
from pipeline.process.process_calendar import process_calendar  # noqa: E402


@pytest.fixture(scope="module")
def df():
    columns = ["date", "week", "winner_tie", "unnamed_5", "loser_tie"]
    data = [
        ["2021-10-17", "6", "Cincinnati Bengals", "@", "Detroit Lions"],
        ["2021-11-28", "12", "New York Jets", "@", "Houston Texans"],
        ["2021-11-01", "8", "Kansas City Chiefs", "", "New York Giants"],
        ["2021-11-21", "11", "Cleveland Browns", "", "Detroit Lions"],
        ["2022-01-09", "18", "Pittsburgh Steelers", "@", "Baltimore Ravens"],
        ["2021-10-31", "8", "San Francisco 49ers", "@", "Chicago Bears"],
    ]
    calendar_df = pd.DataFrame(data, columns=columns)
    return calendar_df


def test_process_calendar(df):
    columns = ["date", "week", "team", "opp", "is_away"]
    expected = pd.DataFrame(
        [
            ["2021-10-17", "6", "CIN", "DET", 1],
            ["2021-10-17", "6", "DET", "CIN", 0],
            ["2021-10-31", "8", "SFO", "CHI", 1],
            ["2021-10-31", "8", "CHI", "SFO", 0],
            ["2021-11-01", "8", "KAN", "NYG", 0],
            ["2021-11-01", "8", "NYG", "KAN", 1],
            ["2021-11-21", "11", "CLE", "DET", 0],
            ["2021-11-21", "11", "DET", "CLE", 1],
            ["2021-11-28", "12", "NYJ", "HOU", 1],
            ["2021-11-28", "12", "HOU", "NYJ", 0],
            ["2022-01-09", "18", "PIT", "BAL", 1],
            ["2022-01-09", "18", "BAL", "PIT", 0],
        ],
        columns=columns,
    )
    print(df)
    result = df.process_calendar()
    assert expected.equals(result)
