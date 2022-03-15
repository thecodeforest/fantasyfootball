import numpy as np
import pandas as pd
import pytest

from fantasyfootball.pipeline.process.process_defense import (
    aggregate_season_defense_stats,
    rank_defense,
    scale_defense_stats,
    weight_defense_stats
)


@pytest.fixture(scope="module")
def df():
    defense_df_dtypes = {
        "opp": object,
        "rushing_yds": float,
        "rushing_td": float,
        "passing_yds": float,
        "passing_td": float,
    }
    data = [
        [
            "TEAM1",
            "TEAM2",
            "TEAM3",
            "TEAM1",
            "TEAM2",
            "TEAM3",
            "TEAM1",
            "TEAM2",
            "TEAM3",
        ],
        [10, 20, 30, 10, 20, 30, 10, 20, 30],
        [1, 2, 3, 1, 2, 3, 1, 2, 3],
        [30, 20, 10, 30, 20, 10, 30, 20, 10],
        [3, 2, 1, 3, 2, 1, 3, 2, 1],
    ]
    defense_df = pd.DataFrame(np.transpose(data), columns=defense_df_dtypes.keys())
    defense_df = defense_df.astype(defense_df_dtypes)
    return defense_df


@pytest.fixture(scope="module")
def aggregated_df(df):
    aggregated_df = (
        df.groupby("opp")
        .agg(
            {
                "rushing_yds": "sum",
                "rushing_td": "sum",
                "passing_yds": "sum",
                "passing_td": "sum",
            }
        )
        .reset_index()
    )
    return aggregated_df


def test_aggregate_season_defense_stats(df, aggregated_df):
    stats_columns = ["rushing_yds", "rushing_td", "passing_yds", "passing_td"]
    expected = aggregated_df
    result = df.aggregate_season_defense_stats(stats_columns)
    assert expected.equals(result)


def test_scale_defense_stats(aggregated_df):
    expected = pd.DataFrame(
        [
            [
                "TEAM1",
                -1.2247448713915892,
                -1.2247448713915892,
                1.2247448713915892,
                1.2247448713915892,
            ],
            ["TEAM2", 0.0, 0.0, 0.0, 0.0],
            [
                "TEAM3",
                1.2247448713915892,
                1.2247448713915892,
                -1.2247448713915892,
                -1.2247448713915892,
            ],
        ],
        columns=aggregated_df.columns,
    )
    result = aggregated_df.scale_defense_stats()
    assert expected.equals(result)


def test_weight_defense_stats(aggregated_df):
    yds_weight = 1  # 100% of weight on yards against, ignoring touchdowns
    expected = pd.DataFrame(
        [
            ["TEAM1", 30.0, 0.0, 90.0, 0.0],
            ["TEAM2", 60.0, 0.0, 60.0, 0.0],
            ["TEAM3", 90.0, 0.0, 30.0, 0.0],
        ],
        columns=aggregated_df.columns,
    )
    result = aggregated_df.weight_defense_stats(yds_weight)
    assert expected.equals(result)


def test_rank_defense(aggregated_df):
    stats_columns = [
        "rushing_yds",
        "passing_yds",
    ]
    expected = pd.DataFrame(
        [["TEAM1", 1.0, 3.0], ["TEAM2", 2.0, 2.0], ["TEAM3", 3.0, 1.0]],
        columns=["opp", "rushing_rank", "passing_rank"],
    )
    result = aggregated_df.rank_defense(stats_columns)
    expected.equals(result)
