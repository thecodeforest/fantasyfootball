from typing import List

import pandas as pd
import pandas_flavor as pf
from sklearn.preprocessing import StandardScaler

from fantasyfootball.config import root_dir
from fantasyfootball.pipeline.utils import (
    get_module_purpose,
    read_args,
    read_ff_csv,
    write_ff_csv
)

DEFENSE_COLUMNS = [
    "rushing_yds",
    "rushing_td",
    "passing_yds",
    "passing_td",
    "receiving_yds",
    "receiving_td",
]


@pf.register_dataframe_method
def aggregate_season_defense_stats(
    df: pd.DataFrame, stats_columns: List[str]
) -> pd.DataFrame:
    """Aggregates the offensive stats for each team up to that week of the season.

    Args:
        df (pd.DataFrame): The dataframe containing the total
            offensive stats against each team.
        stats_columns (List[str]): The columns containing the offensive stats.

    Returns:
        pd.DataFrame: The cumulative offensive stats against each team's defense.
    """
    stats_columns_agg = dict(zip(stats_columns, ["sum"] * len(stats_columns)))
    cumulative_season_stats = df.groupby("opp").agg(stats_columns_agg).reset_index()
    return cumulative_season_stats


@pf.register_dataframe_method
def scale_defense_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Scales the cumulative defensive stats for each team.
       Team stats are scaled in order to incorporate both touchdowns and yards against,
       as these stats are on different scales.

    Args:
        df (pd.DataFrame): The dataframe containing the cumulative
            offensive stats against each team.

    Returns:
        pd.DataFrame: The scaled offensive stats against each team's defense.
    """
    scaler = StandardScaler()
    stats_df = df.select_dtypes(include=float)
    id_df = df.select_dtypes(include=object)
    scaled_stats_mat = scaler.fit_transform(stats_df)
    scaled_stats_df = pd.DataFrame(scaled_stats_mat, columns=stats_df.columns)
    scaled_stats_df = pd.concat([id_df, scaled_stats_df], axis=1)
    return scaled_stats_df


@pf.register_dataframe_method
def weight_defense_stats(df: pd.DataFrame, yds_weight: float) -> pd.DataFrame:
    """Combines the scaled offensive stats against each team's defense. Only
       yards against and touchdowns are currently considered. For example,
       to estimate a defense's strength against the run, `rushing_yds` and
       `rushing_td` are combined. `yds_weight` to determines how much weight
       is given to `rushing_yds` and `rushing_td`.



    Args:
        df (pd.DataFrame): The dataframe containing the scaled offensive
            stats against each team.
        yds_weight (float): How much weight (0 - 1) to give yards against.

    Returns:
        pd.DataFrame: The weighted offensive stats against each team's defense.
    """
    if yds_weight < 0 or yds_weight > 1:
        raise ValueError("`yds_weight` must be between 0 and 1.")
    td_weight = 1 - yds_weight
    td_fields = [x for x in df.columns if x.endswith("_td")]
    yds_fields = [x for x in df.columns if x.endswith("_yds")]
    id_fields = list(set(df.columns) - set(td_fields + yds_fields))
    defense_df = pd.concat(
        [
            df[id_fields],
            df[td_fields].apply(lambda x: x * td_weight),
            df[yds_fields].apply(lambda x: x * yds_weight),
        ],
        axis=1,
    )
    # ensure columns are in original order
    defense_df = defense_df[df.columns]
    return defense_df


@pf.register_dataframe_method
def rank_defense(df: pd.DataFrame, stats_columns: list[str]) -> pd.DataFrame:
    """Creates a rank (1 = Best, 32 = Worst) for each team based on the
       cumulative offensive stats.For example, the team with the most rushing yards
       against the defense will have a `rushing_rank` of 32, while the team with
       the fewest rushing yards against the defense will have a `rushing_rank` of 1.

    Args:
        df (pd.DataFrame): The dataframe containing the cumulative offensive stats
            against each team.
        stats_columns (list[str]): The columns containing the offensive stats.

    Returns:
        pd.DataFrame: The ranked offensive stats against each team's defense.
    """
    df = df.copy()
    point_source = set([x.split("_")[0] for x in stats_columns])
    for points_type in point_source:
        df[f"{points_type}_score"] = df[
            [x for x in df.columns if x.startswith(points_type)]
        ].sum(axis=1)
        df[f"{points_type}_rank"] = df[f"{points_type}_score"].rank()
    df = df[["opp"] + [x for x in df.columns if x.endswith("_rank")]]
    return df


if __name__ == "__main__":
    args = read_args()
    dir_type, data_type = get_module_purpose(module_path=__file__)
    data_dir = root_dir / "data" / "season" / str(args.season_year) / dir_type
    stats_df = read_ff_csv(data_dir / "stats")
    cal_df = read_ff_csv(data_dir / "calendar")
    # add week field to player stats
    processed_stats_df = pd.merge(
        stats_df, cal_df, on=["date", "tm", "opp"], how="inner"
    )
    defense_stats_df = pd.DataFrame()
    for week in sorted(processed_stats_df["week"].unique()):
        cumulative_stats = (
            processed_stats_df.query(f"week <= {week}")
            .aggregate_season_defense_stats(stats_columns=DEFENSE_COLUMNS)
            .scale_defense_stats()
            .weight_defense_stats(yds_weight=0.75)
            .rank_defense()
        )
        cumulative_stats.insert(0, "week", week)
        defense_stats_df = pd.concat([defense_stats_df, cumulative_stats])
    defense_stats_df.write_ff_csv(root_dir, args.season_year, dir_type, data_type)
