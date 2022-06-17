import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import pandera as pa


sys.path.append(str(Path.cwd()))
from pipeline.pipeline_config import root_dir  # noqa: E402
from pipeline.utils import (  # noqa: E402
    get_module_purpose,
    read_args,
    read_ff_csv,
    TEAM_ABBREVIATION_MAPPING,
)


def validate_calendar():
    pass


if __name__ == "__main__":
    TEAM_ABBREVIATION_MAPPING.values()
    args = read_args()
    dir_type, data_type = get_module_purpose(module_path=__file__)
    processed_data_dir = (
        root_dir
        / "staging_datasets"
        / "season"
        / str(args.season_year)
        / "processed"
        / data_type
    )
    processed_calendar_df = read_ff_csv(processed_data_dir)
    processed_calendar_df["date"] = pd.to_datetime(processed_calendar_df["date"])
    start_dt = datetime.strptime(f"{args.season_year}-09-01", "%Y-%m-%d")
    end_dt = datetime.strptime(f"{args.season_year + 1}-01-31", "%Y-%m-%d")
    schema = schema = pa.DataFrameSchema(
        {
            "date": pa.Column(pa.DateTime, checks=pa.Check.in_range(start_dt, end_dt)),
            "week": pa.Column(pa.Int, checks=pa.Check.in_range(1, 18)),
            "team": pa.Column(
                pa.String, checks=pa.Check.isin(TEAM_ABBREVIATION_MAPPING.values())
            ),
            "is_away": pa.Column(pa.Int, checks=pa.Check.in_range(0, 1)),
            "season_year": pa.Column(pa.Int, checks=pa.Check.eq(args.season_year)),
        }
    )
    validated_df = schema(processed_calendar_df)
