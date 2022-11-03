import sys
from pathlib import Path
from itertools import chain
from typing import List

import numpy as np
import pandas as pd
import pandas_flavor as pf
from janitor import clean_names
import awswrangler as wr

sys.path.append(str(Path.cwd()))
from pipeline.pipeline_logger import logger  # noqa: E402
from pipeline.pipeline_config import root_dir  # noqa E402
from pipeline.utils import (  # noqa: E402
    get_module_purpose,
    fetch_current_week,
    read_args,
    read_ff_csv,
    retrieve_team_abbreviation,
    check_if_data_exists,
)


def filter_out_tv_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out rows indicating the network for the game (e.g., FOX, CBS, etc.)

    Args:
        df (pd.DataFrame): The unfiltered dataframe

    Returns:
        pd.DataFrame: The filtered dataframe
    """
    channel_values = [
        "TV:",
        "TV: ",
        "Prime Video",
        "CBS",
        "FOX",
        "NBC",
        "ESPN",
        "Deportes",
        "DEPORTES",
    ]
    df["team"] = df["team"].apply(lambda x: x if x[:4] != "TV: " else x[4:])
    df["team"] = df["team"].apply(lambda x: x if x[:4] != "ESPN" else x[4:])
    for value in channel_values:
        df = df.query(f"team.str.contains('{value}') == False")
    return df


def remove_leading_team_abbr(team_name: str) -> str:
    """Some team names will have an abbreivation as well as the team's name
       (e.g., WASCommanders, TBBuccaneers).
       Need to remove either 2-letter or 3-letter abbreviation in
       order to ]retrieve consistent team mapping.

    Args:
          team_name (str): The unprocessed team name (PatriotsPatriots4-4 (1-2 Home),
          WASCommanders4-4 (2-2 Home), etc.)

    Returns:
          str: The processed team name (Patriots, Commanders, etc.)
    """
    capital_indices = [i for i, letter in enumerate(team_name) if letter.isupper()]
    # find the index that is not consecutive
    for i, index in enumerate(capital_indices):
        if i != 0:
            if index != capital_indices[i - 1] + 1:
                break
    capital_indices = capital_indices[:i]
    # start the name from the last conseuctive index
    team_name = team_name[capital_indices[-1] :]  # noqa: E203
    return team_name


def _find_first_number_index(string: str) -> int:
    "DolphinsDolphins3-0-0 -> 15"
    for i, c in enumerate(string):
        if c.isdigit():
            return i


def _find_second_capital_index(string: str) -> int:
    "DolphinsDolphins -> 8"
    for i, c in enumerate(string):
        if c.isupper() and i != 0:
            return i


def extract_team_name(teamname: str) -> str:
    "DolphinsDolphins3-0-0 -> Dolphins"
    # exception for 49ers
    if "49ers" in teamname:
        return "49ers"
    p1 = teamname[: _find_first_number_index(teamname)]
    p2 = p1[: _find_second_capital_index(p1)]
    return p2


if __name__ == "__main__":
    args = read_args()
    dir_type, data_type = get_module_purpose(module_path=__file__)
    s3_io_path = f"s3://{args.s3_bucket}/datasets/season/{args.season_year}/processed/{data_type}"  # noqa: E501
    calendar_df = pd.read_csv(
        root_dir
        / "staging_datasets"
        / "season"
        / str(args.season_year)
        / "processed"
        / "calendar"
        / "calendar.csv"
    )
    current_week = fetch_current_week(calendar_df)
    raw_data_dir = Path(
        root_dir
        / "staging_datasets"
        / "season"
        / str(args.season_year)
        / "raw"
        / data_type
    )
    betting_df = read_ff_csv(raw_data_dir)
    betting_df = filter_out_tv_rows(betting_df)
    betting_df = betting_df[
        betting_df["team"].str.contains("Home")
        | betting_df["team"].str.contains("Away")
    ]
    row_start, row_end = (0, 2)
    clean_betting_df = pd.DataFrame(
        columns=["team", "opp", "projected_off_pts", "date", "season_year"]
    )
    while row_end <= len(betting_df):
        game_df = betting_df.iloc[row_start:row_end]
        team1, team2 = game_df["team"]
        team1 = remove_leading_team_abbr(team1)
        team2 = remove_leading_team_abbr(team2)
        team1 = extract_team_name(team1)
        team2 = extract_team_name(team2)
        team1_abbr = retrieve_team_abbreviation(team1)
        team2_abbr = retrieve_team_abbreviation(team2)
        # incorrect duplicate values are present, so
        # some teams will appear twice but are incorrectly matched.
        if (
            team1_abbr in clean_betting_df["team"].unique()
            or team2_abbr in clean_betting_df["team"].unique()
        ):
            break
        # over-under
        over_under = round(
            float(game_df["total_points"].iloc[0].split("-")[0].split(" ")[1])
        )
        # game-spread
        team1_spread = game_df["spread"].iloc[0]
        if team1_spread[0] == "+":
            pt_diff = round(float(team1_spread.split("-")[0].replace("+", "")) / 2)
            team1_projected_pts = round((over_under / 2) - pt_diff)
            team2_projected_pts = round((over_under / 2) + pt_diff)
        elif team1_spread[0] == "-":
            pt_diff = round(float(team1_spread.split("-")[1]) / 2)
            team1_projected_pts = round((over_under / 2) + pt_diff)
            team2_projected_pts = round((over_under / 2) - pt_diff)
        else:
            raise ValueError("Spread value not recognized")
        try:
            game_dt = calendar_df.query(
                f"week == {current_week} and team == '{team1_abbr}'"
            )["date"].iloc[0]
        except IndexError:
            # this is a bye week
            logger.info(f"Bye week for {team1_abbr}. Not adding to betting dataset.")
            continue
        team_1_row = [
            team1_abbr,
            team2_abbr,
            team1_projected_pts,
            game_dt,
            args.season_year,
        ]
        team_2_row = [
            team2_abbr,
            team1_abbr,
            team2_projected_pts,
            game_dt,
            args.season_year,
        ]
        game_projections_df = pd.DataFrame(
            [team_1_row, team_2_row],
            columns=["team", "opp", "projected_off_pts", "date", "season_year"],
        )
        clean_betting_df = pd.concat([clean_betting_df, game_projections_df])
        row_start += 2
        row_end += 2
    if wr.s3.list_objects(s3_io_path):
        existing_data = wr.s3.read_csv(f"{s3_io_path}/{data_type}.csv")
        data_exists = check_if_data_exists(
            new_data=clean_betting_df, existing_data=existing_data, time_column="date"
        )
        if data_exists:
            existing_data.write_ff_csv(root_dir, args.season_year, dir_type, data_type)
        else:
            clean_salary_df = pd.concat([existing_data, clean_betting_df])
            clean_salary_df.write_ff_csv(
                root_dir, args.season_year, dir_type, data_type
            )
    else:
        clean_betting_df.write_ff_csv(root_dir, args.season_year, dir_type, data_type)
