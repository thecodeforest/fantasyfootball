from datetime import datetime
import sys
import os
from pathlib import Path

import pandas as pd
import pandera as pa
from pandera.errors import SchemaErrors
import awswrangler as wr

sys.path.append(str(Path.cwd()))
from pipeline.pipeline_config import root_dir  # noqa: E402
from pipeline.pipeline_logger import logger  # noqa: E402
from pipeline.utils import (  # noqa: E402
    read_args,
    read_ff_csv,
    TEAM_ABBREVIATION_MAPPING,
)


def ff_validate(
    df: pd.DataFrame, schema: pa.DataFrameSchema, data_type: str, s3_bucket: str = None
) -> pd.DataFrame:
    try:
        schema.validate(df, lazy=True)
        logger.info(f"Validation of {data_type} dataframe passed.")
        return df
    except SchemaErrors as err:
        logger.info(f"Validation of {data_type} dataframe failed.")
        failure_df = err.failure_cases
        # truncate 'check' field to include up to max of 50 characters
        failure_df["check"] = failure_df["check"].apply(lambda x: x[:30])
        if s3_bucket:
            creation_date = datetime.now().strftime("%Y_%m_%d")
            s3path = (
                f"s3://{s3_bucket}/validation/{creation_date}/{data_type}_errors.csv"
            )
            wr.s3.to_csv(failure_df, s3path, index=False)
        return failure_df


# supplementary datasets
args = read_args()
stadiums_df = pd.read_csv(root_dir / "staging_datasets" / "stadiums.csv")
players_df = pd.read_csv(
    root_dir
    / "staging_datasets"
    / "season"
    / str(args.season_year)
    / "processed"
    / "players"
    / "players.csv"
)
#
players_list = players_df.name.unique()
stadiums_list = stadiums_df.stadium_name.unique()

season_year = args.season_year
# for start and end dates
start_dt = datetime.strptime(f"{season_year}-09-01", "%Y-%m-%d")
end_dt = datetime.strptime(f"{season_year + 1}-01-31", "%Y-%m-%d")
# for list of players in a season
schemas = {
    "calendar": pa.DataFrameSchema(
        {
            "date": pa.Column(pa.DateTime, checks=pa.Check.in_range(start_dt, end_dt)),
            "week": pa.Column(pa.Int, checks=pa.Check.in_range(1, 18)),
            "team": pa.Column(
                pa.String, checks=pa.Check.isin(TEAM_ABBREVIATION_MAPPING.values())
            ),
            "is_away": pa.Column(pa.Int, checks=pa.Check.in_range(0, 1)),
            "season_year": pa.Column(pa.Int, checks=pa.Check.eq(args.season_year)),
        },
        coerce=True,
    ),
    "stats": pa.DataFrameSchema(
        {
            "pid": pa.Column(
                pa.String(),
            ),
            "name": pa.Column(
                pa.String(),
            ),
            "team": pa.Column(
                pa.String(), checks=pa.Check.isin(TEAM_ABBREVIATION_MAPPING.values())
            ),
            "opp": pa.Column(
                pa.String(), checks=pa.Check.isin(TEAM_ABBREVIATION_MAPPING.values())
            ),
            "is_active": pa.Column(pa.Int(), checks=pa.Check.isin([0, 1])),
            "date": pa.Column(pa.DateTime, checks=pa.Check.in_range(start_dt, end_dt)),
            "result": pa.Column(
                pa.String,
                checks=pa.Check(
                    lambda x: x.startswith("W")
                    or x.startswith("L")
                    or x.startswith("T"),
                    element_wise=True,
                ),
            ),
            "is_away": pa.Column(pa.Int(), checks=pa.Check.in_range(0, 1)),
            "is_start": pa.Column(pa.Int(), checks=pa.Check.in_range(0, 1)),
            "g_nbr": pa.Column(pa.Int(), checks=pa.Check.in_range(0, 18)),
            "receiving_rec": pa.Column(
                pa.Float(), checks=pa.Check.greater_than_or_equal_to(0)
            ),
            "receiving_yds": pa.Column(pa.Float()),
            "receiving_td": pa.Column(
                pa.Float(), checks=pa.Check.greater_than_or_equal_to(0)
            ),
            "rushing_yds": pa.Column(pa.Float()),
            "rushing_td": pa.Column(
                pa.Float(), checks=pa.Check.greater_than_or_equal_to(0)
            ),
            "passing_cmp": pa.Column(
                pa.Float(), checks=pa.Check.greater_than_or_equal_to(0)
            ),
            "passing_yds": pa.Column(pa.Float()),
            "passing_td": pa.Column(
                pa.Float(), checks=pa.Check.greater_than_or_equal_to(0)
            ),
            "fumbles_fmb": pa.Column(
                pa.Float(), checks=pa.Check.greater_than_or_equal_to(0)
            ),
            "passing_int": pa.Column(
                pa.Float(), checks=pa.Check.greater_than_or_equal_to(0)
            ),
            "scoring_2pm": pa.Column(
                pa.Float(), checks=pa.Check.greater_than_or_equal_to(0)
            ),
            "punt_returns_td": pa.Column(
                pa.Float(), checks=pa.Check.greater_than_or_equal_to(0)
            ),
        },
        coerce=True,
    ),
    "players": pa.DataFrameSchema(
        {
            "name": pa.Column(pa.String()),
            "team": pa.Column(
                pa.String(), checks=pa.Check.isin(TEAM_ABBREVIATION_MAPPING.values())
            ),
            "position": pa.Column(
                pa.String(), checks=pa.Check.isin(["QB", "RB", "WR", "TE"])
            ),
            "season_year": pa.Column(pa.Int(), checks=pa.Check.eq(args.season_year)),
        },
        coerce=True,
    ),
    "salary": pa.DataFrameSchema(
        {
            "name": pa.Column(pa.String(), checks=pa.Check.isin(players_list)),
            "team": pa.Column(
                pa.String(), checks=pa.Check.isin(TEAM_ABBREVIATION_MAPPING.values())
            ),
            "position": pa.Column(
                pa.String(), checks=pa.Check.isin(["QB", "RB", "WR", "TE"])
            ),
            "season_year": pa.Column(pa.Int(), checks=pa.Check.eq(args.season_year)),
        },
        coerce=True,
    ),
    "weather": pa.DataFrameSchema(
        {
            "date": pa.Column(pa.DateTime, checks=pa.Check.in_range(start_dt, end_dt)),
            "team": pa.Column(
                pa.String(), checks=pa.Check.isin(TEAM_ABBREVIATION_MAPPING.values())
            ),
            "opp": pa.Column(
                pa.String(), checks=pa.Check.isin(TEAM_ABBREVIATION_MAPPING.values())
            ),
            "stadium_name": pa.Column(
                pa.String(), checks=pa.Check.isin(stadiums_df.stadium_name.unique())
            ),
            "is_outdoor": pa.Column(pa.Int(), checks=pa.Check.isin([0, 1])),
            "avg_windspeed": pa.Column(pa.Float(), checks=pa.Check.in_range(0, 100)),
            "max_snow_depth": pa.Column(pa.Float(), checks=pa.Check.in_range(0, 100)),
            "total_precip": pa.Column(pa.Float(), checks=pa.Check.in_range(0, 100)),
            "avg_temp": pa.Column(pa.Float(), checks=pa.Check.in_range(-100, 100)),
        },
        coerce=True,
    ),
    "injury": pa.DataFrameSchema(
        {
            "name": pa.Column(pa.String(), checks=pa.Check.isin(players_list)),
            "team": pa.Column(
                pa.String, checks=pa.Check.isin(TEAM_ABBREVIATION_MAPPING.values())
            ),
            "position": pa.Column(
                pa.String(), checks=pa.Check.isin(["QB", "RB", "WR", "TE"])
            ),
            "season_year": pa.Column(pa.Int(), checks=pa.Check.eq(args.season_year)),
            "week": pa.Column(pa.Int(), checks=pa.Check.in_range(1, 18)),
            "injury_type": pa.Column(
                pa.String(),
            ),
            "has_dnp_tag": pa.Column(pa.Float(), checks=pa.Check.in_range(0, 1)),
            "has_limited_tag": pa.Column(pa.Float(), checks=pa.Check.in_range(0, 1)),
            "most_recent_injury_status": pa.Column(
                pa.String(),
                checks=pa.Check.isin(["no injury", "Limited", "Full", "DNP"]),
            ),
            "n_injuries": pa.Column(
                pa.Float(), checks=pa.Check.greater_than_or_equal_to(0)
            ),
        }
    ),
    "defense": pa.DataFrameSchema(
        {
            "week": pa.Column(pa.Int(), checks=pa.Check.in_range(1, 18)),
            "opp": pa.Column(
                pa.String(), checks=pa.Check.isin(TEAM_ABBREVIATION_MAPPING.values())
            ),
            "rushing_def_rank": pa.Column(
                pa.Float(), checks=pa.Check.in_range(1.0, 32.0)
            ),
            "receiving_def_rank": pa.Column(
                pa.Float(), checks=pa.Check.in_range(1, 32)
            ),
            "passing_def_rank": pa.Column(pa.Float(), checks=pa.Check.in_range(1, 32)),
            "season_year": pa.Column(pa.Int(), checks=pa.Check.eq(args.season_year)),
        },
        coerce=True,
    ),
    "betting": pa.DataFrameSchema(
        {
            "team": pa.Column(
                pa.String(), checks=pa.Check.isin(TEAM_ABBREVIATION_MAPPING.values())
            ),
            "opp": pa.Column(
                pa.String(), checks=pa.Check.isin(TEAM_ABBREVIATION_MAPPING.values())
            ),
            "projected_off_pts": pa.Column(
                pa.Float(), checks=pa.Check.greater_than_or_equal_to(0)
            ),
            "date": pa.Column(pa.DateTime, checks=pa.Check.in_range(start_dt, end_dt)),
            "season_year": pa.Column(pa.Int(), checks=pa.Check.eq(args.season_year)),
        },
        coerce=True,
    ),
}

if __name__ == "__main__":
    for data_type, schema in schemas.items():
        processed_data_dir = (
            root_dir
            / "staging_datasets"
            / "season"
            / str(args.season_year)
            / "processed"
            / data_type
        )
        if Path.exists(processed_data_dir):
            processed_df = read_ff_csv(processed_data_dir)
            ff_validate(
                df=processed_df,
                schema=schema,
                data_type=data_type,
                s3_bucket=args.s3_bucket,
            )
