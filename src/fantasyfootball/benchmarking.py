from itertools import product
import pandas as pd
import pandas_flavor as pf

from fantasyfootball.data import FantasyData
from fantasyfootball.config import root_dir, scoring
from urllib.error import URLError


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
def score_benchmark_data(
    benchmark_df: pd.DataFrame, scoring_source: str
) -> pd.DataFrame:
    """Add point projection based on predictions from:
    https://fantasydata.com/nfl/fantasy-football-weekly-projections
    for use in benchmarking.

    Args:
        benchmark_df (pd.DataFrame): Weekly player predictions from fantasydata.com.
        scoring_source (str): Name of the scoring system to apply (e.g., 'yahoo').

    Returns:
        pd.DataFrame: Weekly player predictions from
            fantasydata converted to scoring system.

    """
    score_player = FantasyData.score_player
    # map different name spellings between
    scoring_source_rules = scoring.get(scoring_source)
    # score all players for that week
    scoring_columns = set(scoring_source_rules["scoring_columns"].keys()) & set(
        benchmark_df.columns
    )
    weekly_benchmark_preds = pd.DataFrame()
    for row in (
        benchmark_df[["name", "team", "position", "season_year"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        player_df = benchmark_df[
            (benchmark_df["name"] == row.name)
            & (benchmark_df["team"] == row.team)
            & (benchmark_df["position"] == row.position)
            & (benchmark_df["season_year"] == row.season_year)
        ]
        player_weekly_points = score_player(
            player_df, scoring_columns, scoring_source_rules
        )
        player_df = player_df.assign(
            **{f"ff_pts_{scoring_source}_fantasydata_pred": player_weekly_points}
        )
        player_df = player_df[
            [
                "name",
                "team",
                "position",
                "season_year",
                "week",
                player_df.columns.tolist()[-1],
            ]
        ]
        weekly_benchmark_preds = pd.concat([weekly_benchmark_preds, player_df])
    return weekly_benchmark_preds


def get_benchmarking_data(
    season_year_start: int,
    season_year_end: int,
    base_url: str = "https://raw.githubusercontent.com/thecodeforest/fantasyfootball/main/examples/benchmarking_data/season/",  # noqa: E501
) -> pd.DataFrame:
    if season_year_start < 2018 or season_year_end > 2021:
        raise ValueError("Season year must be between 2018 and 2021.")
    all_benchmark_df = pd.DataFrame()
    for week, year in product(
        range(1, 18), range(season_year_start, season_year_end + 1)
    ):
        benchmark_url = f"{base_url}/{year}/wk{week}.csv"
        try:
            benchmark_df = pd.read_csv(benchmark_url)
            benchmark_df["season_year"] = year
            all_benchmark_df = pd.concat([all_benchmark_df, benchmark_df])
            return all_benchmark_df
        except URLError:
            raise URLError(
                f"Could not find {benchmark_url}"
                "Check if internet connection is working."
            )
