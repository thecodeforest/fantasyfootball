import pytest
import pandas as pd
from fantasyfootball.benchmarking import (
    get_benchmarking_data,
    filter_to_prior_week,
    score_benchmark_data,
)


@pytest.fixture(scope="module")
def df():
    columns = [
        "date",
        "week",
        "name",
        "team",
        "position",
        "season_year",
        "passing_td",
        "passing_yds",
        "passing_int",
        "rushing_td",
        "rushing_yds",
        "receiving_rec",
        "receiving_td",
        "receiving_yds",
        "fumbles_fmb",
        "scoring_2pm",
        "punt_returns_td",
    ]
    data = [
        [
            "2021-09-09",
            1,
            "Tom Brady",
            "TAM",
            "QB",
            2021,
            4.0,
            379.0,
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            "2021-09-19",
            2,
            "Tom Brady",
            "TAM",
            "QB",
            2021,
            5.0,
            276.0,
            0.0,
            0.0,
            6.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        ],
        [
            "2021-09-26",
            3,
            "Tom Brady",
            "TAM",
            "QB",
            2021,
            1.0,
            432.0,
            0.0,
            1.0,
            14.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        ],
        [
            "2021-10-03",
            4,
            "Tom Brady",
            "TAM",
            "QB",
            2021,
            0.0,
            269.0,
            0.0,
            0.0,
            3.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            "2021-10-10",
            5,
            "Tom Brady",
            "TAM",
            "QB",
            2021,
            5.0,
            411.0,
            0.0,
            0.0,
            13.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            "2021-10-14",
            6,
            "Tom Brady",
            "TAM",
            "QB",
            2021,
            2.0,
            297.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            "2021-10-24",
            7,
            "Tom Brady",
            "TAM",
            "QB",
            2021,
            4.0,
            211.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            "2021-10-31",
            8,
            "Tom Brady",
            "TAM",
            "QB",
            2021,
            4.0,
            375.0,
            2.0,
            0.0,
            2.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        ],
        [
            "2021-11-14",
            10,
            "Tom Brady",
            "TAM",
            "QB",
            2021,
            2.0,
            220.0,
            2.0,
            0.0,
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            "2021-11-22",
            11,
            "Tom Brady",
            "TAM",
            "QB",
            2021,
            2.0,
            307.0,
            1.0,
            0.0,
            10.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            "2021-11-28",
            12,
            "Tom Brady",
            "TAM",
            "QB",
            2021,
            1.0,
            226.0,
            1.0,
            0.0,
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            "2021-12-05",
            13,
            "Tom Brady",
            "TAM",
            "QB",
            2021,
            4.0,
            368.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            "2021-12-12",
            14,
            "Tom Brady",
            "TAM",
            "QB",
            2021,
            2.0,
            363.0,
            0.0,
            1.0,
            16.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            "2021-12-19",
            15,
            "Tom Brady",
            "TAM",
            "QB",
            2021,
            0.0,
            214.0,
            1.0,
            0.0,
            2.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        ],
        [
            "2021-12-26",
            16,
            "Tom Brady",
            "TAM",
            "QB",
            2021,
            1.0,
            232.0,
            0.0,
            0.0,
            11.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            "2022-01-02",
            17,
            "Tom Brady",
            "TAM",
            "QB",
            2021,
            3.0,
            410.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    ]
    benchmarking_df = pd.DataFrame(data, columns=columns)
    return benchmarking_df


def test_score_benchmark_data(df):
    scoring_source = "yahoo"
    expected_col_name = f"ff_pts_{scoring_source}_fantasydata_pred"
    expected_pts = 359
    result = df.score_benchmark_data(scoring_source)
    # ensure column name is correct
    assert result.columns[-1] == expected_col_name
    # ensure points are are correct
    assert result.iloc[0][expected_col_name] == pytest.approx(expected_pts, 1)


def test_filter_to_prior_week(df):
    expected = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14]
    season_year = 2021
    week = 14
    result = df.filter_to_prior_week(season_year, week)
    assert result["week"].tolist() == expected


def test_get_benchmarking_data():
    season_year_start = 2018  # min season start is 2015
    season_year_end = 2021
    result = get_benchmarking_data(
        season_year_start=season_year_start, season_year_end=season_year_end
    )
    assert not result.empty


def test_get_benchmarking_data_incorrect_year():
    season_year_start = 2015  # min season start is 2018
    season_year_end = 2020
    with pytest.raises(ValueError):
        get_benchmarking_data(
            season_year_start=season_year_start, season_year_end=season_year_end
        )
