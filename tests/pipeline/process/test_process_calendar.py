import pandas as pd
import pytest

from fantasyfootball.pipeline.process.process_calendar import process_calendar


@pytest.fixture(scope="module")
def df():
    columns = ["date", "week", "winner_tie", "loser_tie"]
    data = [
        ["2021-09-09", "1", "Tampa Bay Buccaneers", "Dallas Cowboys"],
        ["2021-09-12", "1", "Philadelphia Eagles", "Atlanta Falcons"],
        ["2021-09-12", "1", "Pittsburgh Steelers", "Buffalo Bills"],
        ["2021-10-17", "6", "Arizona Cardinals", "Cleveland Browns"],
        ["2021-10-17", "6", "Kansas City Chiefs", "Washington Football Team"],
        ["2022-01-16", "WildCard", "San Francisco 49ers", "Dallas Cowboys"],
        ["2022-01-16", "WildCard", "Kansas City Chiefs", "Pittsburgh Steelers"],
        ["2022-01-17", "WildCard", "Los Angeles Rams", "Arizona Cardinals"],
        ["2022-01-22", "Division", "Cincinnati Bengals", "Tennessee Titans"],
    ]
    calendar_df = pd.DataFrame(data, columns=columns)
    return calendar_df


def test_process_calendar(df):
    columns = ["date", "week", "tm", "opp"]
    expected = pd.DataFrame(
        [
            ["2021-09-09", "1", "TAM", "DAL"],
            ["2021-09-09", "1", "DAL", "TAM"],
            ["2021-09-12", "1", "PHI", "ATL"],
            ["2021-09-12", "1", "PIT", "BUF"],
            ["2021-09-12", "1", "ATL", "PHI"],
            ["2021-09-12", "1", "BUF", "PIT"],
            ["2021-10-17", "6", "ARI", "CLE"],
            ["2021-10-17", "6", "KAN", "WAS"],
            ["2021-10-17", "6", "CLE", "ARI"],
            ["2021-10-17", "6", "WAS", "KAN"],
        ],
        columns=columns,
    )
    result = df.process_calendar()
    assert expected.equals(result)
