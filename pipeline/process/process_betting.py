import sys
from pathlib import Path
from itertools import chain
from typing import List

import numpy as np
import pandas as pd
import pandas_flavor as pf
from janitor import clean_names

sys.path.append(str(Path.cwd()))
from pipeline.pipeline_config import root_dir  # noqa E402
from pipeline.utils import (  # noqa: E402
    get_module_purpose,
    read_args,
    read_ff_csv,
    retrieve_team_abbreviation,
)


def clean_game_date(season_year: int, date: str) -> str:
    """Creates a date string from a season year and date string.

    Args:
        season_year (int): The season year.
        date (str): The date string.

    Returns:
        str: The date string if the date is part of the regular season, otherwise
            returns 'non-regular-season'.
    """
    if len(date) == 3 and date[0] == "9":
        return f"{season_year}-09-{date[1:]}"
    elif len(date) == 4 and int(date[:2]) > 9:
        return f"{season_year}-{date[:2]}-{date[2:]}"
    elif len(date) == 3 and date[0] in ["1", "2"]:
        season_year += 1
        return f"{season_year}-0{date[0]}-{date[1:]}"
    else:
        return "non-regular-season"


@pf.register_dataframe_method
def clean_games_date(
    df: pd.DataFrame, season_year: int, date_column: str = "date"
) -> pd.DataFrame:
    df[date_column] = (
        df[date_column].astype(str).apply(lambda x: clean_game_date(season_year, x))
    )
    return df


# create game id
@pf.register_dataframe_method
def create_game_id(df: pd.DataFrame) -> pd.DataFrame:
    """Create a unique id for each game. Assumes every two rows are a single game.

    Args:
        df (pd.DataFrame): Dataframe with betting lines data.

    Returns:
        pd.DataFrame: Dataframe with unique id for
        each game that links to the betting lines data.

    Raises:
        ValueError: Occurs when the dataframe has an odd number of rows.
                    An odd row count indicates that there is an incomplete game.
    """
    if divmod(df.shape[0], 2)[1] != 0:
        raise ValueError("Dataframe must have an even number of rows.")
    _id = list(range(1, (len(df) // 2) + 1))
    df["game_id"] = list(chain(*zip(_id, _id)))
    return df


@pf.register_dataframe_method
def add_team_abbreviation(df: pd.DataFrame, team_column: str = "team") -> pd.DataFrame:
    """Convert team name to team abbreviation.

    Args:
        df (pd.DataFrame): Dataframe with betting lines data and team column.
        team_column (str, optional): Column with full team name. Defaults to "team".

    Returns:
        pd.DataFrame: Dataframe with team abbreviation column.
    """
    df[team_column] = df[team_column].apply(lambda x: retrieve_team_abbreviation(x))
    return df


@pf.register_dataframe_method
def create_point_spread_df(df: pd.DataFrame):
    """Convert over-under (O/U) and line to projected points for each team.
    For example, if the O/U is 49 and Team 1 is favored by -7 over Team 2,
    the point projection for Team 1 is 28 and 21 for Team 2.

    Args:
        df (pd.DataFrame): Dataframe with betting lines data.

    Returns:
        pd.DataFrame: Dataframe with point spread data.
    """
    is_even_moneyline = df[df["ml"] < 0].shape[0] > 1
    is_pick = any(df["open"].str.contains("pk"))
    if any([is_even_moneyline, is_pick]):
        fav_team, underdog_team = df["team"]
        fav_pts, underdog_pts = [
            float(max([x for x in df["open"] if x != "pk"])) / 2
        ] * 2
    else:
        fav_team_index = [index for index, value in enumerate(df["ml"]) if value < 0][0]
        underdog_team_index = int(not fav_team_index)
        fav_team = df["team"].iloc[fav_team_index]
        underdog_team = df["team"].iloc[underdog_team_index]
        pt_spread = df["open"].astype(float).min()
        over_under = df["open"].astype(float).max()
        fav_pts = (over_under / 2) + pt_spread * 0.5
        underdog_pts = (over_under / 2) - pt_spread * 0.5
    spread_df = pd.DataFrame(
        [[fav_team, underdog_team, fav_pts], [underdog_team, fav_team, underdog_pts]],
        columns=["team", "opp", "projected_off_pts"],
    )
    spread_df["projected_off_pts"] = spread_df["projected_off_pts"].apply(
        lambda x: round(x)
    )
    return spread_df


@pf.register_dataframe_method
def process_betting(df: pd.DataFrame, season_year: int) -> pd.DataFrame:
    """Converts raw betting lines data to include point projections for each
    team in each game. Point projections can be used to inform how much
    scoring is expected to occur in each game.

    Args:
        df (pd.DataFrame): Raw betting lines data.
        season_year (int): The season year.

    Returns:
        pd.DataFrame: Dataframe with point projections for each team in each game.
    """
    process_betting_df = pd.DataFrame()
    game_ids = df["game_id"].unique()
    for game_id in game_ids:
        game_df = df[df["game_id"] == game_id]
        point_spread_df = game_df.create_point_spread_df()
        point_spread_df["date"] = game_df["date"].iloc[0]
        point_spread_df["season_year"] = season_year
        process_betting_df = pd.concat([process_betting_df, point_spread_df])
    return process_betting_df


def impute_missing_projections(
    df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    keys: List[str] = ["date", "season_year", "team", "opp"],
    default_point_projection: int = 25,
) -> pd.DataFrame:
    """Impute missing point projections for each team in each game.
       Some dates for betting are incorrect. In these instances,
       the point projections are imputed with the average point projection
       for all previous games played in the same season.

    Args:
        df (pd.DataFrame): Dataframe with point projections for each team in each game.
        calendar_df (pd.DataFrame): Dataframe with dates for each game.
        keys (List[str], optional): Columns to join the point projections and calendar.
            Defaults to ["date", "season_year", "team", "opp"].
        default_point_projection (int, optional): If a team has no
            previous games (i.e., it's the first
            game of the season), the point projection is imputed with this value.
            Defaults to 25.

    Returns:
        pd.DataFrame: Dataframe with imputed point projections for each team in
            each game. There should be no NA values for the point projections.
    """
    # identify games without a projection value
    missing_projection_df = pd.merge(calendar_df, df, on=keys, how="left")
    if not missing_projection_df["projected_off_pts"].isnull().sum():
        return pd.DataFrame()
    missing_projection_df = missing_projection_df[
        pd.isna(missing_projection_df["projected_off_pts"])
    ]
    imputed_projections_lst = list()
    for row in missing_projection_df.itertuples(index=False):
        date, season_year, team, opp = row.date, row.season_year, row.team, row.opp
        # filter prior projections and averge together for imputed projection
        prior_week_projections_df = df[(df["date"] < date) & (df["team"] == team)]
        if prior_week_projections_df.shape[0] == 0:
            print(
                "No historical data to calculate projection."
                f"Using default projection of {default_point_projection}"
            )
            avg_week_projections = default_point_projection
        else:
            avg_week_projections = round(
                np.average(prior_week_projections_df["projected_off_pts"])
            )
        imputed_projections_lst.append(
            [date, season_year, team, opp, avg_week_projections]
        )
    imputed_projections_df = pd.DataFrame(
        imputed_projections_lst, columns=keys + ["projected_off_pts"]
    )
    return imputed_projections_df


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
    clean_betting_df = read_ff_csv(raw_data_dir)
    calendar_df = pd.read_csv(
        root_dir
        / "staging_datasets"
        / "season"
        / str(args.season_year)
        / "processed"
        / "calendar"
        / "calendar.csv"
    )
    clean_betting_df = (
        clean_betting_df.clean_names()
        .clean_games_date(args.season_year)
        .create_game_id()
        .add_team_abbreviation()
        .process_betting(args.season_year)
    )
    imputed_projection_df = impute_missing_projections(clean_betting_df, calendar_df)
    if not imputed_projection_df.empty:
        clean_betting_df = pd.concat([clean_betting_df, imputed_projection_df])
    clean_betting_df.write_ff_csv(root_dir, args.season_year, dir_type, data_type)
