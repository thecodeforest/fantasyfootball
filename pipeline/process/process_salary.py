"""url: https://dailyroto.com/nfl-historical-production\
        -fantasy-points-draftkings-fanduel
"""
import sys
from pathlib import Path

import pandas as pd
import pandas_flavor as pf
from janitor import clean_names, coalesce
import awswrangler as wr

sys.path.append(str(Path.cwd()))
from pipeline.pipeline_config import root_dir  # noqa: E402
from pipeline.utils import (  # noqa: E402
    get_module_purpose,
    map_abbr2_to_abbr3,
    map_player_names,
    read_args,
    read_ff_csv,
)


def _name_format(player_name: str) -> str:
    player_name = player_name.split(", ")
    player_name = player_name[-1] + " " + "".join(player_name[:-1])
    return player_name


@pf.register_dataframe_method
def process_salary(df: pd.DataFrame) -> pd.DataFrame:
    # select relevant fields
    df = df[["player", "pos", "year", "week", "salary"]]
    # rename to consistent column names
    df = df.rename(
        columns={
            "player": "name",
            "pos": "position",
            "year": "season_year",
            "salary": "fanduel_salary",
        }
    )
    # filter only to relevant positions (QB, RB, WR, TE)
    df = df[df["position"].isin(["QB", "RB", "WR", "TE"])]
    # remove dollar sign from salary and convert to int
    df["fanduel_salary"] = (
        df["fanduel_salary"].str.replace("$", "", regex=True).astype(int)
    )
    # convert season_year & week to int
    df["season_year"] = df["season_year"].astype(int)
    df["week"] = df["week"].astype(int)
    # format player name by switching position of first and last name
    df["name"] = df["name"].apply(lambda x: _name_format(x))
    return df


if __name__ == "__main__":
    args = read_args()
    dir_type, data_type = get_module_purpose(module_path=__file__)
    s3_io_path = f"s3://{args.s3_bucket}/datasets/season/{args.season_year}/processed/{data_type}"  # noqa: E501
    players_df = pd.read_csv(
        root_dir
        / "staging_datasets"
        / "season"
        / str(args.season_year)
        / "processed"
        / "players"
        / "players.csv"
    )

    raw_data_dir = Path(
        root_dir
        / "staging_datasets"
        / "season"
        / str(args.season_year)
        / "raw"
        / data_type
    )

    clean_salary_df = read_ff_csv(raw_data_dir)
    clean_salary_df = clean_salary_df.clean_names()
    clean_salary_df = clean_salary_df.process_salary()
    clean_salary_df = (
        pd.merge(
            clean_salary_df,
            map_player_names(players_df, clean_salary_df, "name", "position"),
            on=["name", "position"],
            how="left",
        )
        .coalesce("mapped_name", "name", target_column_name="final_name")
        .drop(columns=["name", "mapped_name"])
        .rename(columns={"final_name": "name"})
    )
    clean_salary_df = clean_salary_df[
        [clean_salary_df.columns.tolist()[-1]] + clean_salary_df.columns.tolist()[:-1]
    ]
    # write this weeks clean data to S3
    week = str(clean_salary_df["week"].iloc[0])
    file_name = f"week_{week}_salary.csv"
    wr.s3.to_csv(clean_salary_df, f"{s3_io_path}/{file_name}", index=False)

    all_files = wr.s3.list_objects(s3_io_path)
    clean_salary_df = wr.s3.read_csv(all_files)
    clean_salary_df.write_ff_csv(root_dir, args.season_year, dir_type, data_type)
