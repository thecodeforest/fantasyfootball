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
    # filter any rows where week is NA
    df = df[~df["week"].isna()]
    # filter only to rows where week is a number
    df = df[[x.isnumeric() for x in df["week"]]]
    # flip positioning of teams so all possible combinations are covered
    df = pd.concat(
        [
            df[["date", "week", "winner_tie", "loser_tie"]],
            df[["date", "week", "loser_tie", "winner_tie"]].rename(
                columns={"winner_tie": "loser_tie", "loser_tie": "winner_tie"}
            ),
        ]
    )
    df["tm"] = df["winner_tie"].apply(lambda x: retrieve_team_abbreviation(x))
    df["opp"] = df["loser_tie"].apply(lambda x: retrieve_team_abbreviation(x))
    df = df.drop(columns=["winner_tie", "loser_tie"])
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
