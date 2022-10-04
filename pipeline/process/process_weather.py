import sys
from pathlib import Path

import pandas as pd
import pandas_flavor as pf

sys.path.append(str(Path.cwd()))
from pipeline.pipeline_config import root_dir  # noqa: E402
from pipeline.utils import (  # noqa: E402
    get_module_purpose,
    read_args,
    read_ff_csv,
    write_ff_csv,
)


@pf.register_dataframe_method
def process_weather_df(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans raw weather data by applying the following transformations:
        * Add feature indicating if the game was played outdoors or indoors
        * Add '0' value for windspeed when game is played indoors or retractable roof
        * Add '0' value for snow when game is played indoors or retractable roof
        * Add '0' value for rain when game is played indoors or retractable roof
        * Assign constant of 75 degrees F when game was played indoors

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df = df.assign(
        **{"temperature": df["temperature"].apply(lambda x: 75 if x == "" else x)}
    )
    # add zero windspeed for indoor games
    df = df.assign(
        **{"wind_speed": df["wind_speed"].apply(lambda x: 0 if x == "" else x)}
    )
    # add zero for is_rain for indoor games
    df = df.assign(**{"is_rain": df["is_rain"].apply(lambda x: 0 if x == "" else x)})
    # add zero for is_snow for indoor games
    df = df.assign(**{"is_snow": df["is_snow"].apply(lambda x: 0 if x == "" else x)})
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce").round(1)
    df["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce").round(1)
    # # if the roof_type is outdoor, assign 1 else 0
    df["is_outdoor"] = df["roof_type"].apply(lambda x: 1 if x == "Outdoor" else 0)
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
