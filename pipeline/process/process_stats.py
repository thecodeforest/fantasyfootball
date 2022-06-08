import sys
from pathlib import Path
import re
from itertools import chain

import pandas as pd
import pandas_flavor as pf
from janitor import clean_names

sys.path.append(str(Path.cwd()))
from pipeline.pipeline_config import root_dir  # noqa: E402
from pipeline.pipeline_logger import logger  # noqa: E402
from pipeline.utils import get_module_purpose, read_args, write_ff_csv  # noqa: E402

REQUIRED_COLUMNS = {
    "player_columns": ["pid", "name"],
    "game_columns": ["tm", "opp", "is_active", "date", "result", "is_away", "is_start"],
    "stats_columns": [
        "g_nbr",
        "receiving_rec",
        "receiving_yds",
        "receiving_td",
        "rushing_yds",
        "rushing_td",
        "passing_cmp",
        "passing_yds",
        "passing_td",
        "fumbles_fmb",
        "passing_int",
        "scoring_2pm",
        "punt_returns_td",
    ],
}


@pf.register_dataframe_method
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the column names of the given dataframe by applying the following steps
       after using the janitor `clean_names` function:
        * strips any 'unnamed' field, for example 'Unnamed: 0'
        * replaces the first missing name with 'is_away'
        * coverts '#' to '_nbr'
        * converts '%' to '_pct'

    Args:
        df (pd.DataFrame): The dataframe to clean the column names of.

    Returns:
        pd.DataFrame: The dataframe with cleaned column names.
    """
    df = clean_names(df)
    cols = df.columns
    cols = [re.sub("unnamed_[0-9]+_level_[0-9]", "", x).strip("_") for x in cols]
    # away will always be the first empty string following cleaning step above
    cols[cols.index("")] = "is_away"
    cols = [x.replace("#", "_nbr") for x in cols]
    cols = [x.replace("%", "_pct") for x in cols]
    cols = ["is_active" if x == "status" else x for x in cols]
    cols = ["is_start" if x == "gs" else x for x in cols]
    df.columns = cols
    return df


@pf.register_dataframe_method
def clean_status_column(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the status column indicating if a player was playing or not.
       If a player did not missing any games during a season,
       the status column will not be present
       and will need to be added to the dataframe.
       An active status is denoted by an empty string. Any
       other status indicates the player was not
       active (e.g., "Did Not Play", "Covid-19", etc.).

    Args:
        df (pd.DataFrame): Dataframe with the status column.

    Returns:
        pd.DataFrame: The Dataframe with the status column cleaned.
    """
    # empty string indicates player played, otherwise player was injured, sick, etc.
    if "is_active" not in df.columns:
        df["is_active"] = [1] * df.shape[0]
    else:
        df["is_active"] = [1 if not x else 0 for x in df["is_active"]]
    return df


@pf.register_dataframe_method
def add_missing_stats_columns(
    df: pd.DataFrame, required_columns: dict = REQUIRED_COLUMNS
) -> pd.DataFrame:
    """Detects missing columns in the dataframe and adds them to the dataframe.
       For example, if a player
       has never thrown a pass, not 'passing' columns will be present
       in the dataframe.

    Args:
        df (pd.DataFrame): The dataframe to add missing stats columns to.
        required_columns (dict, optional): Indicates which
                                           player stats columns must be included.
        Defaults to REQUIRED_COLUMNS indicated at the top of this file.

    Raises:
        ValueError: If a "non-stats" column is missing
                    (e.g., "pid", "name", "team", "opp",etc.),
                    an error is raised because filling with zeros is not appropriate.

    Returns:
        pd.DataFrame: The dataframe with missing stats columns added.
    """
    df = df.copy()
    current_columns = df.columns.tolist()
    missing_columns = set(chain(*required_columns.values())) - set(current_columns)
    non_stats_columns = set(
        chain(
            *[
                required_columns[x]
                for x in required_columns.keys()
                if x != "stats_columns"
            ]
        )
    )
    # check for overlap between missing_columns and non_stats_columns
    missing_non_stats_columns = missing_columns & non_stats_columns
    if missing_non_stats_columns:
        raise ValueError(
            f"Columns {missing_non_stats_columns} are missing but not stats columns."
        )
    if missing_columns:
        for column in missing_columns:
            df[column] = 0
    return df


@pf.register_dataframe_method
def select_ff_columns(
    df: pd.DataFrame, required_columns: dict = REQUIRED_COLUMNS
) -> pd.DataFrame:
    """Selects the columns that are required for the fantasy football data.

    Args:
        df (pd.DataFrame): The dataframe to select the required columns from.
        required_columns (dict, optional): Indicates which player stats columns
                                           must be included. Defaults
                                           to REQUIRED_COLUMNS
                                           indicated at the top of this file.

    Returns:
        pd.DataFrame: The dataframe with the required columns selected.
    """
    return df[
        required_columns["player_columns"]
        + required_columns["game_columns"]
        + required_columns["stats_columns"]
    ]


def clean_stats_column(col: pd.Series) -> list:
    """Cleans the column values of the given series by applying the following steps:
        * Fills NA with "0"
        * Strips percent signs from the end of the string
        * Replaces any non-numeric values with "0"
        * Converts all values to a float

    Args:
        col (pd.Series): The series to clean the column values of.

    Returns:
        list: The cleaned column values.
    """
    col = col.fillna("0")
    col = col.astype(str).tolist()
    col = [x.strip("%") for x in col]
    numeric_values = [x.replace(".", "").isdigit() for x in col]
    if not all(numeric_values):
        non_numeric_value_indexes = [
            index for (index, value) in enumerate(numeric_values) if not value
        ]
        for i in non_numeric_value_indexes:
            col[i] = "0"
    col = [float(x) for x in col]
    return col


@pf.register_dataframe_method
def clean_stats_column_values(
    df: pd.DataFrame, stats_columns: list = REQUIRED_COLUMNS["stats_columns"]
) -> pd.DataFrame:
    df = df.copy()
    for col in stats_columns:
        df[col] = clean_stats_column(col=df[col])
    return df


@pf.register_dataframe_method
def recode_str_to_numeric(
    df: pd.DataFrame, column: str, target_value: str, replacement_value: int
) -> pd.DataFrame:
    """Replaces the target_value with the replacement_value (1, 0) in the given column.

    Args:
        df (pd.DataFrame): The dataframe to recode the column values of.
        column (str): The column to recode the values of.
        target_value (str): The value to replace.
        replacement_value (int): The value to replace the target_value with.
                                 Must be 1 or 0.

    Raises:
        ValueError: If the replacement_value is not 1 or 0, an error is raised.

    Returns:
        pd.DataFrame: The dataframe with the column values recoded.
    """
    if replacement_value not in [0, 1]:
        raise ValueError("replacement_value must be 0 or 1")
    other_replacement_value = int(not replacement_value)
    df = df.assign(
        **{
            column: df[column].transform(
                lambda x: replacement_value
                if x == target_value
                else other_replacement_value
            )
        }
    )
    return df


if __name__ == "__main__":
    args = read_args()
    dir_type, data_type = get_module_purpose(module_path=__file__)
    raw_data_dir = (
        root_dir
        / "staging_datasets"
        / "season"
        / str(args.season_year)
        / "raw"
        / data_type
    )
    raw_stats_files = raw_data_dir.glob("*.csv")
    for player_stats_path in raw_stats_files:
        clean_stats_df = pd.read_csv(player_stats_path, keep_default_na=False)
        pid = clean_stats_df["pid"].iloc[0]
        logger.info(f"Processing player {pid}")
        clean_stats_df = (
            clean_stats_df.clean_column_names()
            .rename(columns={"gs": "is_start"})
            .clean_status_column()
            .add_missing_stats_columns()
            .select_ff_columns()
            .clean_stats_column_values()
            .query("~date.str.contains('Games')")
            .recode_str_to_numeric(
                column="is_away", target_value="@", replacement_value=1
            )
            .recode_str_to_numeric(
                column="is_start", target_value="*", replacement_value=1
            )
            .rename(columns={"tm": "team"})
        )
        clean_stats_df.write_ff_csv(
            root_dir, args.season_year, dir_type, data_type, pid
        )
