import sys
from pathlib import Path

import pandas as pd
import pandas_flavor as pf

sys.path.append(str(Path.cwd()))
from config import root_dir  # noqa: E402
from utils import (  # noqa: E402
    get_module_purpose,
    read_args,
    read_ff_csv,
    write_ff_csv,
)


@pf.register_dataframe_method
def process_weather_df(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans raw weather data by applying the following transformations:
        * Add feature indicating if the game was played outdoors or indoors
        * Convert windspeed from km to mph
        * Add '0' value for windspeed when game is played indoors
        * Add feature indicating total amount of snow (inches)
        * Add feature indicating total amount of rain (inches)
        * Convert temperature from F to C
        * Assign constant of 75 degrees F when game was played indoors

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df = df.drop(columns=["tmin", "tmax", "wdir", "wpgt", "pres", "tsun"])
    # add dome feature
    df = df.assign(
        **{"is_outdoor": df["roof_type"].apply(lambda x: 1 if x == "Outdoor" else 0)}
    ).drop("roof_type", axis=1)
    # convert wind speed to mph
    df = df.assign(
        **{
            "avg_windspeed": df["wspd"].apply(
                lambda x: 0 if x == "" else float(x) * 2.237
            )
        }
    ).drop("wspd", axis=1)
    # convert mm to inches for snow
    df = df.assign(
        **{
            "max_snow_depth": df["snow"].apply(
                lambda x: 0 if x == "" else float(x) * 0.0393701
            )
        }
    ).drop("snow", axis=1)
    # convert total daily precip to inches
    df = df.assign(
        **{
            "total_precip": df["prcp"].apply(
                lambda x: 0 if x == "" else float(x) * 0.0393701
            )
        }
    ).drop("prcp", axis=1)
    # convert average temp to fahrenheit
    avg_temp_for_dome = 75
    df = df.assign(
        **{
            "avg_temp": df["tavg"].apply(
                lambda x: avg_temp_for_dome if x == "" else float(x) * 1.8 + 32
            )
        }
    ).drop("tavg", axis=1)
    return df


if __name__ == "__main__":
    args = read_args()
    dir_type, data_type = get_module_purpose(module_path=__file__)
    raw_data_dir = (
        root_dir
        / "staging_datasets"
        / "season"
        / str(args.season_year)
        / "raw"
        / data_type
    )
    clean_weather_df = read_ff_csv(raw_data_dir)
    clean_weather_df = clean_weather_df.process_weather_df()
    clean_weather_df.write_ff_csv(root_dir, args.season_year, dir_type, data_type)
