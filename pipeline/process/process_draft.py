from socket import if_indextoname
import sys
from pathlib import Path

import pandas as pd
from janitor import clean_names

sys.path.append(str(Path.cwd()))
from pipeline.pipeline_config import root_dir  # noqa: E402
from pipeline.utils import (  # noqa: E402
    get_module_purpose,
    map_player_names,
    retrieve_team_abbreviation,
    read_args,
    read_ff_csv,
    map_abbr2_to_abbr3,
)


def process_draft(df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
    # remove defenses from the draft
    df = df[~df["team"].apply(lambda x: x.replace("(", "").replace(")", "").isdigit())]
    # filter out free agents
    df = df[df["team"] != "FA"]
    # translate team abbreviations to 3 letter
    df["team"] = df["team"].apply(lambda x: map_abbr2_to_abbr3(x))
    # map player names
    df = (
        pd.merge(
            df,
            map_player_names(players_df, df, "name", "team"),
            on=["name", "team"],
            how="left",
        )
        .coalesce("mapped_name", "name", target_column_name="final_name")
        .drop(columns=["name", "mapped_name"])
        .rename(columns={"final_name": "name"})
    )
    # add in player position
    df = pd.merge(players_df, df, on=["name", "team"], how="left")
    # filter out players that are not QB, RB, WR, TE
    df = df[df["position"].isin(["QB", "RB", "WR", "TE"])]
    # bump avg_draft_positon by 1
    df["avg_draft_position"] = df["avg_draft_position"] + 1
    # convert avg_draft_position to int
    df["avg_draft_position"] = df["avg_draft_position"].astype(float)
    # for undrafted players, set avg_draft_position to max avg_draft_position + 1
    df["avg_draft_position"] = df["avg_draft_position"].fillna(
        df["avg_draft_position"].max() + 1
    )
    df["avg_draft_position"] = df["avg_draft_position"].astype(int)
    # rearrange column order
    df = df[["avg_draft_position", "name", "team", "position", "season_year"]]
    # sort by avg_draft_position
    df = df.sort_values(by="avg_draft_position", ascending=True).reset_index(drop=True)
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
    clean_draft_df = read_ff_csv(raw_data_dir)
    clean_draft_df = clean_names(clean_draft_df)
    clean_draft_df = process_draft(clean_draft_df, players_df)
    clean_draft_df.write_ff_csv(root_dir, args.season_year, dir_type, data_type)
