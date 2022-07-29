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


def fill_undrafted_player_positions(
    df: pd.DataFrame,
    players_df: pd.DataFrame,
    avg_draft_position_column: str = "avg_draft_position",
) -> pd.DataFrame:
    """Replace undrafted players with the maximum drafted position + 1 to ensure no NAs

    Args:
        df (pd.DataFrame): Dataframe with all of the players who were drafted
        players_df (pd.DataFrame): Dataframe with all of the
            players who were active within a season
        avg_draft_position_column (str, optional): Column indicating a player's
            average draft position. Defaults to "avg_draft_position".

    Returns:
        pd.DataFrame: Dataframe with undrafted players filled
            in with the maximum drafted position + 1
    """
    df = df.copy()
    # join draft data w/ active players to identify undrafted players
    df = pd.merge(players_df, df, on=["name", "team", "position"], how="left")
    # find maximum draft position
    max_draft_position = df[avg_draft_position_column].max() + 1
    # replace any missing values with the maximum draft position
    df[avg_draft_position_column] = df[avg_draft_position_column].fillna(
        max_draft_position
    )
    return df


def process_draft(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"pos": "position", "overall": "avg_draft_position"})
    df = df[["name", "position", "team", "avg_draft_position"]]
    df = df[df["position"].isin(["QB", "RB", "WR", "TE"])]
    # filter out players who are free agents
    df = df[df["team"] != "FA"]
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
    df = fill_undrafted_player_positions(df, players_df, "avg_draft_position")
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
