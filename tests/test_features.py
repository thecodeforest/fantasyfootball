import numpy as np
import pandas as pd
import pytest

from fantasyfootball.features import FantasyFeatures


@pytest.fixture(scope="module")
def df():
    columns = [
        "date",
        "week",
        "team",
        "opp",
        "is_away",
        "season_year",
        "name",
        "pid",
        "position",
        "projected_pts",
        "rushing_def_rank",
        "receiving_def_rank",
        "passing_def_rank",
        "rushing_yds",
        "passing_yds",
        "receiving_yds",
        "is_outdoor",
        "avg_temp",
    ]
    data = [
        [
            "2021-10-14",
            6,
            "TAM",
            "PHI",
            1,
            2021,
            "Tom Brady",
            "BradTo00",
            "QB",
            30.0,
            27.0,
            10.0,
            13.0,
            1.0,
            297.0,
            0.0,
            1,
            69.44,
        ],
        [
            "2021-10-24",
            7,
            "TAM",
            "CHI",
            0,
            2021,
            "Tom Brady",
            "BradTo00",
            "QB",
            30.0,
            21.0,
            13.0,
            11.0,
            0.0,
            211.0,
            0.0,
            1,
            80.78,
        ],
        [
            "2021-10-31",
            8,
            "TAM",
            "NOR",
            1,
            2021,
            "Tom Brady",
            "BradTo00",
            "QB",
            27.0,
            1.0,
            11.0,
            14.0,
            2.0,
            375.0,
            0.0,
            0,
            75.0,
        ],
        [
            "2021-10-17",
            6,
            "DAL",
            "NWE",
            1,
            2021,
            "Dak Prescott",
            "PresDa01",
            "QB",
            24.0,
            7.0,
            18.0,
            20.0,
            10.0,
            445.0,
            0.0,
            1,
            59.0,
        ],
        [
            "2021-10-31",
            8,
            "DAL",
            "MIN",
            1,
            2021,
            "Dak Prescott",
            "PresDa01",
            "QB",
            24.0,
            18.0,
            9.0,
            12.0,
            0.0,
            0.0,
            0.0,
            1,
            41.72,
        ],
        [
            "2021-10-31",
            8,
            "NYJ",
            "CIN",
            0,
            2021,
            "Joe Flacco",
            "FlacJo00",
            "QB",
            21.0,
            7.0,
            21.0,
            24.0,
            0.0,
            0.0,
            0.0,
            1,
            58.64,
        ],
        [
            "2021-10-17",
            6,
            "GNB",
            "CHI",
            1,
            2021,
            "Aaron Rodgers",
            "RodgAa00",
            "QB",
            25.0,
            15.0,
            9.0,
            8.0,
            19.0,
            195.0,
            0.0,
            1,
            54.5,
        ],
        [
            "2021-10-24",
            7,
            "GNB",
            "WAS",
            0,
            2021,
            "Aaron Rodgers",
            "RodgAa00",
            "QB",
            29.0,
            15.0,
            32.0,
            32.0,
            17.0,
            274.0,
            0.0,
            1,
            39.38,
        ],
        [
            "2021-10-28",
            8,
            "GNB",
            "ARI",
            1,
            2021,
            "Aaron Rodgers",
            "RodgAa00",
            "QB",
            25.0,
            19.0,
            4.0,
            9.0,
            3.0,
            184.0,
            0.0,
            0,
            75.0,
        ],
    ]
    historical_df = pd.DataFrame(data, columns=columns)
    return historical_df


def test_filter_n_games_played_by_season(df):
    # Remove Joe Flacco, who has only played 1 game during this time period.
    expected_players = 3
    min_games_played = 2
    features = FantasyFeatures(df, position="QB")
    features.filter_n_games_played_by_season(min_games_played=min_games_played)
    result_players = len(features.data["pid"].unique())
    assert result_players == expected_players


@pytest.mark.parametrize(
    "columns, row_values, expected",
    [
        (["col1", "col2"], [1, "z"], 'col1==1 and col2=="z"'),
        (["col1"], [1], "col1==1"),
        (["col2"], ["a"], 'col2=="a"'),
    ],
)
def test__format_pd_query(columns, row_values, expected):
    _format_pd_filter_query = FantasyFeatures._format_pd_filter_query
    assert _format_pd_filter_query(columns, row_values) == expected


@pytest.mark.parametrize(
    "columns, feature_type, values, expected",
    [
        (["passing_td"], "lag", 1, ["passing_td_lag_1"]),
        (["passing_td"], "ma", 4, ["passing_td_ma_4"]),
        (["rushing_td"], "avg", "", ["rushing_td_avg"]),
    ],
)
def test__save_pipeline_feature_names(columns, feature_type, values, expected):
    _save_pipeline_feature_names = FantasyFeatures._save_pipeline_feature_names
    assert _save_pipeline_feature_names(columns, feature_type, values) == expected


@pytest.mark.parametrize(
    "step, transformer_name, kwargs, expected",
    [
        (
            "Description of Transformation",
            "FeatureTransformer",
            {"kwargs1": "test"},
            "('Description of Transformation', FeatureTransformer(kwargs1=test))",
        ),
    ],
)
def test__create_step_str(step, transformer_name, kwargs, expected):
    _create_step_str = FantasyFeatures._create_step_str
    assert _create_step_str(step, transformer_name, **kwargs) == expected
