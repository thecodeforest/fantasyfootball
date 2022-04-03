import pandas as pd
import pandas_flavor as pf

from fantasyfootball.config import root_dir


@pf.register_dataframe_method
def filter_to_prior_week(
    df: pd.DataFrame, season_year: int, week_number: int
) -> pd.DataFrame:
    calendar_df = pd.read_csv(
        root_dir.parent.parent
        / "datasets"
        / "season"
        / str(season_year)
        / "calendar.csv"
    )
    prior_week_df = calendar_df[calendar_df["week"] == week_number]
    max_date_week = max(prior_week_df["date"])
    prior_week_df = df[df["date"] <= max_date_week]
    return prior_week_df
