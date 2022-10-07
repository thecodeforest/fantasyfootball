import pandas as pd
import numpy as np
import pytest
from fantasyfootball.config import data_sources, root_dir
from fantasyfootball.features import (
    FantasyFeatures,
    CategoryConsolidatorFeatureTransformer,
    TargetEncoderFeatureTransformer,
)


@pytest.fixture(scope="module")
def df():
    columns = [
        "date",
        "week",
        "team",
        "opp",
        "is_away",
        "is_active",
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
        "injury_type",
        "actual_pts",
    ]
    data = [
        [
            "2021-10-07",
            5,
            "TAM",
            "CIN",
            1,
            0,
            2021,
            "Tom Brady",
            "BradTo00",
            "QB",
            0,
            0,
            0,
            0,
            0,
            0,
            0.0,
            0,
            0,
            "head",
            0,
        ],
        [
            "2021-10-14",
            6,
            "TAM",
            "PHI",
            1,
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
            "no injury",
            1,
        ],
        [
            "2021-10-24",
            7,
            "TAM",
            "CHI",
            0,
            1,
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
            "hip",
            2,
        ],
        [
            "2021-10-31",
            8,
            "TAM",
            "NOR",
            1,
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
            "hip",
            2,
        ],
        [
            "2021-10-17",
            6,
            "DAL",
            "NWE",
            1,
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
            "head",
            3,
        ],
        [
            "2021-10-31",
            8,
            "DAL",
            "MIN",
            1,
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
            "no injury",
            1,
        ],
        [
            "2021-10-31",
            8,
            "NYJ",
            "CIN",
            0,
            1,
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
            "no injury",
            1,
        ],
        [
            "2021-10-17",
            6,
            "GNB",
            "CHI",
            1,
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
            "no injury",
            1,
        ],
        [
            "2021-10-24",
            7,
            "GNB",
            "WAS",
            0,
            1,
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
            "hip",
            2,
        ],
        [
            "2021-10-28",
            8,
            "GNB",
            "ARI",
            1,
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
            "no injury",
            1,
        ],
    ]
    historical_df = pd.DataFrame(data, columns=columns)
    return historical_df


def test_filter_n_games_played_by_season(df):
    # Remove Joe Flacco, who has only played 1 game during this time period.
    expected_players = 3
    min_games_played = 2
    features = FantasyFeatures(df, y="actual_pts", position="QB")
    features.filter_n_games_played_by_season(min_games_played=min_games_played)
    result_players = len(features.data["pid"].unique())
    assert result_players == expected_players


def test_filter_inactive_games(df):
    # Remove week 5 for Tom Brady where he is inactive and drop the `is_active` column
    status_column = "is_active"
    expected_rows = df.shape[0] - 1
    features = FantasyFeatures(df, y="actual_pts", position="QB")
    features.filter_inactive_games(status_column=status_column)
    result_rows = features.data.shape[0]
    # verify that week 5 dropped
    assert result_rows == expected_rows
    # verify that the `is_active` column is dropped
    assert status_column not in features.data.columns


def test__calculate_n_games_played(df):
    expected = [3, 1, 2, 3]
    player_group_columns = ["pid", "name", "team", "season_year"]
    calculate_n_games_played = FantasyFeatures._calculate_n_games_played
    result = calculate_n_games_played(df, player_group_columns)
    assert result["n_games_played"].tolist() == expected


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


def test_CategoryConsolidatorFeatureTransformer(df):
    category_column = "injury_type"
    threshold = 0.3
    expected = [
        "other",  # changed from 'head' to 'other', given it appears below the threshold
        "no injury",
        "other",  # hip
        "other",  # hip
        "other",  # changed from 'head' to 'other', given it appears below the threshold
        "no injury",
        "no injury",
        "no injury",
        "other",  # hip
        "no injury",
    ]
    cc = CategoryConsolidatorFeatureTransformer(
        category_columns=category_column, threshold=threshold
    )
    result = cc.fit_transform(df)[category_column].tolist()
    assert result == expected


def test_TargetEncoderFeatureTransformer(df):
    category_column = "injury_type"
    te_category_column = f"{category_column}_te"
    expected = [1.5, 1.0, 2.0, 2.0, 1.5, 1.0, 1.0, 1.0, 2.0, 1.0]
    te = TargetEncoderFeatureTransformer(category_columns="injury_type")
    X = df[df.columns.tolist()[:-1]]
    y = df[df.columns.tolist()[-1]]
    result = te.fit_transform(X, y)[te_category_column].tolist()
    assert result == expected


def test__validate_future_data_is_present():
    expected = True
    season_year = 2021
    max_week = 16
    ff_data_dir = root_dir / "datasets" / "season" / str(season_year)
    _validate_future_data_is_present = FantasyFeatures._validate_future_data_is_present
    result = _validate_future_data_is_present(ff_data_dir, max_week, data_sources)
    assert result == expected


def test__validate_future_data_is_present_error():
    season_year = 2021
    max_week = 18
    ff_data_dir = root_dir / "datasets" / "season" / str(season_year)
    _validate_future_data_is_present = FantasyFeatures._validate_future_data_is_present
    with pytest.raises(ValueError):
        assert _validate_future_data_is_present(ff_data_dir, max_week, data_sources)


def test_add_coefficient_of_variation(df):
    expected = 47.0
    features = FantasyFeatures(df, y="actual_pts", position="QB")
    features.add_coefficient_of_variation(n_week_window=2)
    # check for Aaron Rodgers most recent value
    result = features.data.query("pid == 'RodgAa00'")["cv"].tolist()[-1]
    assert expected == result


def test__validate_column_present(df):
    column = "passing_yds"
    expected = True
    features = FantasyFeatures(df, y="actual_pts", position="QB")
    result = features._validate_column_present(column)
    assert result == expected


def test__validate_column_present_error(df):
    column = "invalid_column_name"
    features = FantasyFeatures(df, y="actual_pts", position="QB")
    with pytest.raises(ValueError):
        features._validate_column_present(column)


def test_add_lag_feature(df):
    lag_columns = "passing_yds"
    n_week_lag = 1
    expected_column_name = "passing_yds_lag_1"
    expected_values = [0.0, 297.0, 211.0, 0, 195.0, 274.0]
    features = FantasyFeatures(df, y="actual_pts", position="QB")
    features.add_lag_feature(n_week_lag=n_week_lag, lag_columns=lag_columns)
    result = features.create_ff_signature().get("feature_df")

    # check column name correctly formatted
    assert expected_column_name in result.columns

    # check column values correct
    assert result[expected_column_name].fillna(0).values.tolist() == expected_values


def test_add_moving_average_feature(df):
    window_columns = "passing_yds"
    n_week_window = 2
    expected_column_name = "passing_yds_ma_2"
    expected_values = expected_values = [
        0,
        0,
        148.5,
        254.0,
        0,
        0,
        445.0,
        0,
        195.0,
        234.5,
    ]
    features = FantasyFeatures(df, y="actual_pts", position="QB")
    features.add_moving_avg_feature(
        n_week_window=n_week_window, window_columns=window_columns
    )
    result = features.create_ff_signature().get("feature_df")

    # check column name correctly formatted
    assert expected_column_name in result.columns

    # check column values correct, also fillna(0) to account for NaNs
    assert result[expected_column_name].fillna(0).values.tolist() == expected_values


def test_add_target_encoded_feature(df):
    category_columns = "injury_type"
    expected_column_name = "injury_type_te"
    expected_values = [1.5, 1.0, 2.0, 2.0, 1.0, 1.5, 1.0, 1.0, 2.0, 1.0]
    features = FantasyFeatures(df, y="actual_pts", position="QB")
    features.add_target_encoded_feature(category_columns=category_columns)
    result = features.create_ff_signature().get("feature_df")

    # check column name correctly formatted
    assert expected_column_name in result.columns

    # check column values correct
    assert result[expected_column_name].values.tolist() == expected_values


def test_add_category_consolidator_feature(df):
    category_columns = "injury_type"
    threshold = 0.2
    expected_values = [
        "other",
        "no injury",
        "hip",
        "hip",
        "no injury",
        "other",
        "no injury",
        "no injury",
        "hip",
        "no injury",
    ]
    features = FantasyFeatures(df, y="actual_pts", position="QB")
    features.consolidate_category_feature(
        category_columns=category_columns, threshold=threshold
    )
    result = features.create_ff_signature().get("feature_df")

    # check column values correct
    assert result[category_columns].values.tolist() == expected_values
