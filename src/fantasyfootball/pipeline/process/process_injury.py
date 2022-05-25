import pandas as pd
import pandas_flavor as pf
from janitor import coalesce
from pandas.api.types import is_numeric_dtype, is_string_dtype

from fantasyfootball.config import root_dir
from fantasyfootball.pipeline.utils import (
    get_module_purpose,
    map_player_names,
    read_args,
    read_ff_csv,
    retrieve_team_abbreviation,
)


@pf.register_dataframe_method
def add_injury_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse each row of injury data by player. Adds the following columns:
        * has_dnp_status: Indicates if the player had a DNP (Did Not Play) status
          during the week.
        * has_limited_status: Indicates if the player had a limited status
          during the week.
        * most_recent_injury_status: The most recently reported injury status.
        * n_injuries: The number of injuries reported for the player.

    Args:
        df (pd.DataFrame): The dataframe of week-player level injury data.

    Returns:
        pd.DataFrame: The original dataframe with the added injury columns.
    """
    injury_features_lst = list()
    for row in df.itertuples(index=False):
        status = (
            row.mon_status,
            row.tue_status,
            row.wed_status,
            row.thu_status,
            row.fri_status,
            row.sat_status,
            row.sun_status,
        )
        # '--' indicates an unknown status. Translate to misssing.
        status = ["" if x == "--" else x for x in status]
        # has DNP tag
        has_dnp_tag = int(any([x for x in status if x == "DNP"]))
        # has limited tag
        has_limited_tag = int(any([x for x in status if x == "Limited"]))
        # empty string indicates
        try:
            most_recent_status = [x for x in status if x][-1]
        except IndexError:
            most_recent_status = ""
        # count number of injuries
        n_injuries = len(row.injury_type.split(","))
        feature_row = [
            row.name,
            row.position,
            row.team,
            row.week,
            row.season_year,
            has_dnp_tag,
            has_limited_tag,
            most_recent_status,
            n_injuries,
        ]
        injury_features_lst.append(feature_row)
    injury_features_df = pd.DataFrame(
        injury_features_lst,
        columns=[
            "name",
            "position",
            "team",
            "week",
            "season_year",
            "has_dnp_tag",
            "has_limited_tag",
            "most_recent_injury_status",
            "n_injuries",
        ],
    )
    df = pd.merge(
        df,
        injury_features_df,
        how="inner",
        on=["name", "position", "team", "week", "season_year"],
    )
    return df


def _convert_plural_injury_to_singular(injury: str) -> str:
    """Converts a plural injury type to a singular injury type.
       For example, 'Broken Arms' becomes 'Broken Arm', or 'Ribs' becomes 'Rib'.

    Args:
        injury (str): The injury type to convert.

    Returns:
        str: The singularized injury type.
    """
    injury_split = list(injury)
    if injury_split[-1] == "s":
        return injury[: len(injury_split) - 1]
    return injury


@pf.register_dataframe_method
def process_injury_type(df: pd.DataFrame, column: str = "injury_type") -> pd.DataFrame:
    """Formats the injury type column by applying the following transformations:
        * Convert all injury text to lower case
        * Strip "left" and "right" from the end of the injury text
        * Convert 'not injury related', 'non football injury',
          and 'load management' to 'non-injury related'
        * If multiple conditions are present, take the first injury reported,
          splitting on comma or slash
        * Convert "abdomen", "core", "stomach" to "abdomen"
        * Convert a plural injury type to a singular injury type

    Args:
        df (pd.DataFrame): The dataframe of week-player level injury data.
        column (str): The column name of the injury type column.
           Defaults to "injury_type".

    Returns:
        pd.DataFrame: The original dataframe with the processed injury type column.

    """
    injury_type = df[column]
    injury_type = ["unknown" if not x else x for x in injury_type]
    injury_type = [x.lower() for x in injury_type]
    injury_type = [
        x.replace("right", "").replace("left", "").strip() for x in injury_type
    ]
    injury_type = [
        "not injury related"
        if "not injury related" in x
        or "non football injury" in x
        or "load management" in x
        else x
        for x in injury_type
    ]
    injury_type = [x.split(",")[0] for x in injury_type]
    injury_type = [x.split("/")[0] for x in injury_type]
    injury_type = [
        "abdomen" if x in ["abdomen", "core", "stomach"] else x for x in injury_type
    ]
    injury_type = [_convert_plural_injury_to_singular(x) for x in injury_type]
    df[column] = injury_type
    return df


@pf.register_dataframe_method
def add_missing_values_for_non_injured_players(
    df: pd.DataFrame, players_df: pd.DataFrame
) -> pd.DataFrame:
    """Identifies players who were not injured during the week by taking the
       cross product of the weekly injuries and all players active within a season.

    Args:
        df (pd.DataFrame): The dataframe of week-player level injury data.
        players_df (pd.DataFrame): The dataframe of all players active within a season.

    Returns:
        pd.DataFrame: All player-week-injury possible combinations.
    """
    min_week, max_week = min(df["week"].astype(int)), max(df["week"].astype(int))
    all_season_weeks_df = pd.DataFrame({"week": list(range(min_week, max_week + 1))})
    cross_product_players_df = pd.merge(players_df, all_season_weeks_df, how="cross")
    df = pd.merge(
        cross_product_players_df,
        df,
        on=cross_product_players_df.columns.tolist(),
        how="left",
    )
    return df


@pf.register_dataframe_method
def fill_missing_values_for_non_injured_players(df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing values for players who were not injured during the week,
       depending on the datatype. Strings are replaced with 'no injury', while
       numeric values are replaced with 0.

    Args:
        df (pd.DataFrame): The dataframe of week-player level injury data.

    Raises:
        ValueError: If the datatype of the column is not a string or numeric.

    Returns:
        pd.DataFrame: The original dataframe with the filled missing values.
    """
    pct_missing_by_column = df.isnull().mean()
    for column, pct_missing in zip(pct_missing_by_column.index, pct_missing_by_column):
        if pct_missing > 0:
            if is_string_dtype(df[column]):
                df[column] = df[column].fillna("no injury")
            elif is_numeric_dtype(df[column]):
                df[column] = df[column].fillna(0)
            else:
                raise ValueError(f"{column} is not a string or numeric type")
    return df


if __name__ == "__main__":
    args = read_args()
    dir_type, data_type = get_module_purpose(module_path=__file__)
    players_df = pd.read_csv(
        root_dir
        / "staging_datasets"
        / "season"
        / str(args.season_year)
        / "processed"
        / "players"
        / "players.csv"
    )
    raw_data_dir = (
        root_dir
        / "staging_datasets"
        / "season"
        / str(args.season_year)
        / "raw"
        / data_type
    )
    clean_injury_df = read_ff_csv(raw_data_dir)
    clean_injury_df = clean_injury_df[
        clean_injury_df["position"].isin(["QB", "RB", "WR", "TE"])
    ]
    clean_injury_df = clean_injury_df[clean_injury_df["team"] != ""]
    clean_injury_df["season_year"] = clean_injury_df["season_year"].astype(int)
    clean_injury_df["week"] = clean_injury_df["week"].astype(int)
    # filter empty team names & map full team name to 3 letter abbreviation
    clean_injury_df["team"] = clean_injury_df["team"].apply(
        lambda x: retrieve_team_abbreviation(x)
    )
    # map name to correct player name
    clean_injury_df = (
        pd.merge(
            clean_injury_df,
            map_player_names(players_df, clean_injury_df, "name", "team", "position"),
            on=["name", "team", "position"],
            how="left",
        )
        .coalesce("mapped_name", "name", target_column_name="final_name")
        .drop(columns=["name", "mapped_name"])
        .rename(columns={"final_name": "name"})
    )
    # create injury features
    clean_injury_df = (
        clean_injury_df.add_injury_feature_columns()
        .process_injury_type()
        .add_missing_values_for_non_injured_players(players_df)
        .fill_missing_values_for_non_injured_players()
    )
    clean_injury_df = clean_injury_df[
        [
            "name",
            "team",
            "position",
            "season_year",
            "week",
            "injury_type",
            "has_dnp_tag",
            "has_limited_tag",
            "most_recent_injury_status",
            "n_injuries",
        ]
    ]
    clean_injury_df.write_ff_csv(root_dir, args.season_year, dir_type, data_type)
