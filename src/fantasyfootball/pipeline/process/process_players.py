import pandas as pd
import pandas_flavor as pf

from fantasyfootball.config import root_dir
from fantasyfootball.pipeline.utils import (
    get_module_purpose,
    read_args,
    read_ff_csv,
    write_ff_csv,
)


def is_valid_team(team: str) -> bool:
    """Filters out players who played for multiple teams, were primarly on
       the practice squad, or were free agents.

    Args:
        team_col (str): Name of the column indicating the player's team
    """
    invalid_team_values = ["", "2TM", "3TM", "4TM"]
    if team in invalid_team_values:
        return False
    else:
        return True


def is_valid_position(position: str) -> bool:
    """Filters out players without a valid position.

    Args:
        position (str): Player's current position

    Returns:
        bool: True if the player has a valid position, False otherwise
    """
    valid_position_values = ["TE", "RB", "WR", "QB"]
    if position not in valid_position_values:
        return False
    else:
        return True


@pf.register_dataframe_method
def process_players(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the dataframe containing the player's name, position, team, and
       season year by applying the following steps:
       * Remove players who played for multiple teams within a season or
         were primarily on the practice squad
       * Remove header rows for original table
       * Remove characters indicating accolades (e.g., probowl)

    Args:
        df (pd.DataFrame): Raw dataframe containing player information

    Returns:
        pd.DataFrame: Clean dataframe containing player information
    """
    df_processed = df.copy()
    # remove rows with invalid team values
    df_processed = df_processed[df_processed["team"].apply(lambda x: is_valid_team(x))]
    # remove rows for players without a position
    df_processed = df_processed[
        df_processed["position"].apply(lambda x: is_valid_position(x))
    ]
    # remove any rows with original headers - Player, team, FantPos, 2021
    df_processed = df_processed[
        df_processed["player"].apply(lambda x: not x.startswith("Player"))
    ]
    # strip *+ indicating pro bowl or other accolades - replace with assign
    df_processed["player"] = df_processed["player"].apply(
        lambda x: x.replace("*", "").replace("+", "").strip()
    )
    # change player column to name to be consistent with stats columns
    df_processed = df_processed.rename(columns={"player": "name"})
    df_processed = df_processed.reset_index(drop=True)
    return df_processed


if __name__ == "__main__":
    args = read_args()
    dir_type, data_type = get_module_purpose(module_path=__file__)
    raw_data_dir = (
        root_dir / "datasets" / "season" / str(args.season_year) / "raw" / data_type
    )
    players_raw = read_ff_csv(raw_data_dir)
    players_processed = players_raw.process_players()
    players_processed.write_ff_csv(
        root_dir=root_dir,
        season_year=args.season_year,
        dir_type=dir_type,
        data_type=data_type,
    )
