import pandas as pd
import pandas_flavor as pf
from janitor import coalesce

from fantasyfootball.config import root_dir, scoring
from fantasyfootball.data import FantasyData
from fantasyfootball.pipeline.utils import map_player_names

_score_player = FantasyData._score_player


@pf.register_dataframe_method
def filter_to_prior_week(
    df: pd.DataFrame, season_year: int, week_number: int
) -> pd.DataFrame:
    """Filter all data up until the most recently
       completed week.


    Args:
        df (pd.DataFrame): Historical data and features.
        season_year (int): Year of the season.
        week_number (int): Week number of the most recently completed week.

    Returns:
        pd.DataFrame: Historical data and features.
    """
    calendar_df = pd.read_csv(
        root_dir / "datasets" / "season" / str(season_year) / "calendar.gz"
    )
    prior_week_df = calendar_df[calendar_df["week"] == week_number]
    max_date_week = max(prior_week_df["date"])
    prior_week_df = df[df["date"] <= max_date_week]
    return prior_week_df


@pf.register_dataframe_method
def process_benchmark_preds(
    benchmark_df: pd.DataFrame, reference_df: pd.DataFrame, yvar: str
) -> pd.DataFrame:
    """Format data from https://fantasydata.com/nfl/fantasy-football-weekly-projections
       for use in benchmarking. The following steps are performed:
         * Map player names from fantasydata.com
           to player names in fantasyfootball package.
         * Apply scoring system to stat projections for each player

    Args:
        benchmark_df (pd.DataFrame): Weekly player predcitions from fantasydata.com.
        reference_df (pd.DataFrame): Player data from fantasyfootball package.
                                     Used for mapping player names.
        yvar (str): Name of the scoring system to apply (e.g., 'yahoo').

    Returns:
        pd.DataFrame: Weekly player predcitions from fantasydata.com.

    """
    # map different name spellings between
    scoring_source_rules = scoring.get(yvar.replace("ff_pts_", ""))
    benchmark_df = (
        pd.merge(
            benchmark_df,
            map_player_names(reference_df, benchmark_df, "name", "team", "position"),
            on=["name", "team", "position"],
            how="left",
        )
        .coalesce("mapped_name", "name", target_column_name="final_name")
        .drop(columns=["name", "mapped_name"])
        .rename(columns={"final_name": "name"})
    )
    # score all players for that week
    scoring_columns = set(scoring_source_rules["scoring_columns"].keys()) & set(
        benchmark_df.columns
    )
    weekly_benchmark_preds = pd.DataFrame()
    for row in (
        benchmark_df[["name", "team", "position"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        player_df = benchmark_df[
            (benchmark_df["name"] == row.name)
            & (benchmark_df["team"] == row.team)
            & (benchmark_df["position"] == row.position)
        ]
        player_weekly_points = _score_player(
            player_df, scoring_columns, scoring_source_rules
        )
        player_df = player_df.assign(
            **{f"{yvar}_fantasydata_pred": player_weekly_points}
        )
        player_df = player_df[
            ["name", "team", "position", "week", player_df.columns.tolist()[-1]]
        ]
        weekly_benchmark_preds = pd.concat([weekly_benchmark_preds, player_df])
    weekly_benchmark_preds = weekly_benchmark_preds.drop_duplicates()
    return weekly_benchmark_preds
