import sys
from pathlib import Path

import pandas as pd
from janitor import clean_names

sys.path.append(str(Path.cwd()))
from pipeline.pipeline_config import root_dir  # noqa: E402
from pipeline.utils import (  # noqa: E402
    get_module_purpose,
    map_player_names,
    read_args,
    read_ff_csv,
    map_abbr2_to_abbr3,
)


def process_draft(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"pos": "position", "overall": "avg_draft_position"})
    df = df[["name", "position", "team", "avg_draft_position"]]
    df = df[df["position"].isin(["QB", "RB", "WR", "TE"])]
    df["team"] = df["team"].apply(lambda x: map_abbr2_to_abbr3(x))
    df = (
        pd.merge(
            df,
            map_player_names(players_df, df, "name", "team", "position"),
            on=["name", "team", "position"],
            how="left",
        )
        .coalesce("mapped_name", "name", target_column_name="final_name")
        .drop(columns=["name", "mapped_name"])
        .rename(columns={"final_name": "name"})
    )
    df = df[[df.columns[-1]] + df.columns[:-1].tolist()]
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
    clean_draft_df = process_draft(clean_draft_df)
    clean_draft_df["season_year"] = args.season_year
    clean_draft_df.write_ff_csv(root_dir, args.season_year, dir_type, data_type)
