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
from pipeline.pipeline_config import root_dir  # noqa E402
from pipeline.utils import (  # noqa: E402
    get_module_purpose,
    fetch_current_week,
    read_args,
    read_ff_csv,
    retrieve_team_abbreviation,
    check_if_data_exists,
)


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
    row_start, row_end = (0, 2)
    clean_betting_df = pd.DataFrame()
    while row_end <= len(betting_df):
        game_df = betting_df.iloc[row_start:row_end]
        team1, team2 = game_df["team"]
        team1_abbr = retrieve_team_abbreviation(extract_team_name(team1))
        team2_abbr = retrieve_team_abbreviation(extract_team_name(team2))
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
        game_dt = calendar_df.query(
            f"week == {current_week} and team == '{team1_abbr}'"
        )["date"].iloc[0]

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
