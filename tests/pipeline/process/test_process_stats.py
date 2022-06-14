import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path.cwd()))
from pipeline.process.process_stats import (  # noqa: E402
    add_missing_stats_columns,
    clean_column_names,
    clean_stats_column,
    clean_status_column,
    recode_str_to_numeric,
)


@pytest.fixture(scope="module")
def df():
    columns = [
        "g_nbr",
        "is_start",
        "passing_yds",
        "passing_cmp_pct",
        "is_away",
        "scoring_2pm",
        "off_snaps_pct",
        "is_active",
    ]
    data = [
        ["1", "*", "250", "75.0%", "", "1", "72%", ""],
        ["2", "*", "275", "55.0%", "@", "", "81%", ""],
        ["3", "*", "320", "66.7%", "@", "", "95%", ""],
        [
            "4",
            "Did Not Play",
            "Did Not Play",
            "Did Not Play",
            "",
            "",
            "Did Not Play",
            "Did Not Play",
        ],
        [
            "5",
            "Did Not Play",
            "Did Not Play",
            "Did Not Play",
            "",
            "",
            "Did Not Play",
            "Did Not Play",
        ],
    ]
    stats_df = pd.DataFrame(data, columns=columns)
    return stats_df


def test_clean_column_names(df):
    """
    Test to make sure that the column names are cleaned correctly
    """
    df_original_columns = df.copy()
    expected = df.columns.tolist()
    original_column_names = [
        "Unnamed: 2_level_0_G#",  # game number
        "Unnamed: 9_level_0_GS",  # games started
        "Passing_Yds",  # passing yards
        "Passing_Cmp%",  # passing completion percent
        "Unnamed: 6_level_0_Unnamed: 6_level_1",  # indicates if game is home or away
        "Scoring_2PM",  # number of 2 point scores
        "Off. Snaps_Pct",  # percent of offensive snaps played
        "Unnamed: 47_level_0_Status",  # indicates if a player was active in the game
    ]
    df_original_columns.columns = original_column_names
    result = clean_column_names(df_original_columns).columns.tolist()
    assert result == expected


def test_add_missing_stats_columns(df):
    missing_columns = ["rushing_yds"]
    required_columns = {"stats_columns": list(df.columns) + missing_columns}
    result = add_missing_stats_columns(df, required_columns=required_columns)
    assert result.columns.tolist() == required_columns["stats_columns"]
    assert sum(result[missing_columns].sum(axis=1)) == 0


def test_add_missing_stats_columns_error(df):
    missing_columns = ["date"]
    required_columns = {
        **{"stats_columns": list(df.columns)},
        **{"game_columns": missing_columns},
    }
    with pytest.raises(ValueError):
        add_missing_stats_columns(df, required_columns=required_columns)


def test_clean_stats_column(df):
    # test passing_yds column
    expected = [250, 275, 320, 0, 0]
    col = "passing_yds"
    result = clean_stats_column(df[col])
    assert result == expected
    # test passing_cmp_pct column
    expected = [75, 55, 66.7, 0, 0]
    col = "passing_cmp_pct"
    result = clean_stats_column(df[col])
    assert result == expected


def test_clean_status_column_present(df):
    expected = [1, 1, 1, 0, 0]
    result = clean_status_column(df)
    assert result["is_active"].tolist() == expected


def test_clean_status_column_absent(df):
    expected = [1, 1, 1, 1, 1]
    df_no_status = df.copy()
    df_no_status = df_no_status.drop(columns="is_active")
    result = clean_status_column(df_no_status)
    assert result["is_active"].tolist() == expected


def test_recode_str_to_numeric(df):
    # test away column
    expected = [0, 1, 1, 0, 0]
    column = "is_away"
    target_value = "@"
    replacement_value = 1
    result = recode_str_to_numeric(df, column, target_value, replacement_value)
    assert result[column].tolist() == expected
    # test games started column
    expected = [1, 1, 1, 0, 0]
    column = "is_start"
    target_value = "*"
    replacement_value = 1
    result = recode_str_to_numeric(df, column, target_value, replacement_value)
    assert result[column].tolist() == expected
