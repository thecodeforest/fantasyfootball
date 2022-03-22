import pandas as pd
import pandas_flavor as pf
from janitor import clean_names

from fantasyfootball.config import root_dir
from fantasyfootball.pipeline.utils import (
    get_module_purpose,
    read_args,
    read_ff_csv,
    retrieve_team_abbreviation,
    write_ff_csv
)


def create_away_team_df(df: pd.DataFrame) -> pd.DataFrame:
    """Create a dataframe to indicate if a team is away or home.

    Args:
        df (pd.DataFrame): The calendar dataframe.

    Returns:
        pd.DataFrame: The dataframe with the away team column added.
    """
    away_game_lst = list()
    for row in df.itertuples():
        if row.away == "@":
            # team on left is away
            away_game_lst.append([row.week, row.date, row.team, 1])
            away_game_lst.append([row.week, row.date, row.opp, 0])
        else:
            away_game_lst.append([row.week, row.date, row.team, 0])
            away_game_lst.append([row.week, row.date, row.opp, 1])
    away_game_df = pd.DataFrame(away_game_lst, columns=["week", "date", "team", "away"])
    return away_game_df


@pf.register_dataframe_method
def process_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the calendar dataframe by applying the following steps:
        * Filter rows where the week is missing
        * Filter rows where week is missing
        * Ensure both teams in a game are represented in the dataframe
        * Add the team abbreviation for each team

    Args:
        df (pd.DataFrame): The dataframe to clean.

    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    # name away field
    df = df.rename(columns={"unnamed_5": "away"})
    # filter any rows where week is NA
    df = df[~df["week"].isna()]
    # filter only to rows where week is a number
    df = df[[x.isnumeric() for x in df["week"]]]
    df["team"] = df["winner_tie"].apply(lambda x: retrieve_team_abbreviation(x))
    df["opp"] = df["loser_tie"].apply(lambda x: retrieve_team_abbreviation(x))
    away_team_df = create_away_team_df(df)
    # flip positioning of teams so all possible combinations are covered
    df = pd.concat(
        [
            df[["date", "week", "team", "opp"]],
            df[["date", "week", "opp", "team"]].rename(
                columns={"opp": "team", "team": "opp"}
            ),
        ]
    )
    df = pd.merge(df, away_team_df, how="inner", on=["week", "date", "team"])
    df = df.sort_values("date")
    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    args = read_args()
    dir_type, data_type = get_module_purpose(module_path=__file__)
    raw_data_dir = (
        root_dir / "data" / "season" / str(args.season_year) / "raw" / data_type
    )
    clean_calendar_df = read_ff_csv(raw_data_dir)
    clean_calendar_df = clean_calendar_df.clean_names().process_calendar()
    clean_calendar_df.write_ff_csv(root_dir, args.season_year, dir_type, data_type)
