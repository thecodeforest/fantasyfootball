import sys
from pathlib import Path
from datetime import datetime, timedelta
from time import sleep
import requests
import json
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd  # noqa: E402
from meteostat import Daily, Point  # noqa: E402

sys.path.append(str(Path.cwd()))
from pipeline.pipeline_config import root_dir  # noqa: E402
from pipeline.pipeline_logger import logger  # noqa: E402
from pipeline.utils import get_module_purpose, read_args, write_ff_csv  # noqa: E402


def filter_calendar_horizon(
    calendar_df: pd.DataFrame, fcast_horizon_days: int = 6
) -> pd.DataFrame:
    """Filter calendar to only include games prior to and within the forecast horizon.

    Args:
        calendar_df (pd.DataFrame): Calendar dataframe.
        fcast_horizon_days (int, optional): How many days into the future to consider
            for weather forecast. Defaults to 6.

    Returns:
        pd.DataFrame: Filtered calendar dataframe with
            only games within the forecast horizon.
    """

    dt_range_end = datetime.today() + timedelta(days=fcast_horizon_days)
    calendar_df = calendar_df[calendar_df["date"] <= dt_range_end.strftime("%Y-%m-%d")]
    return calendar_df


def collect_daily_weather_historicals(
    game_date: str, latitude: float, longitude: float
) -> pd.DataFrame:
    """Collect historical weather conditions from Meteostat API.

    Args:
        game_date (str): Game date.
        latitude (float): Stadium latitude.
        longitude (float): Stadium longitude.

    Returns:
        pd.DataFrame: Historical weather conditions dataframe.
    """
    game_date = datetime.strptime(game_date, "%Y-%m-%d")
    point = Point(lat=latitude, lon=longitude)
    df = Daily(point, start=game_date, end=game_date).fetch().reset_index()
    df = df[["time", "tavg", "prcp", "snow", "wspd"]]
    df = df.rename(
        columns={
            "time": "date",
            "tavg": "temperature",
            "wspd": "wind_speed",
            "prcp": "is_rain",
            "snow": "is_snow",
        }
    )
    # conver temperate from celsius to fahrenheit
    df["temperature"] = df["temperature"] * 9 / 5 + 32
    # convert km/h to mph
    df["wind_speed"] = df["wind_speed"] * 0.621371
    # convert nan in is_snow to 0
    df["is_snow"] = df["is_snow"].fillna(0)
    # if is_rain greater than 0, set to 1
    df["is_rain"] = df["is_rain"].apply(lambda x: 1 if x > 0 else 0)
    # if is_snow greater than 0, set to 1
    df["is_snow"] = df["is_snow"].apply(lambda x: 1 if x > 0 else 0)
    # convert date to string
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def collect_daily_weather_forecast(
    latitude: float, longitude: float, api_key: str
) -> pd.DataFrame:
    """Collect weather forecast from OpenWeatherMap API.

    Args:
        latitude (float): Stadium latitude.
        longitude (float): Stadium longitude.
        api_key (str): OpenWeatherMap API key.

    Returns:
        pd.DataFrame: Weather forecast dataframe.
    """
    forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={latitude}&lon={longitude}&units=imperial&appid={api_key}"  # noqa: E501
    data = requests.get(forecast_url).json()
    columns = [
        "datetime",
        "temp",
        "feels_like",
        "temp_min",
        "temp_max",
        "pressure",
        "humidity",
        "weather_main",
        "weather_description",
        "wind_speed",
        "wind_gust",
    ]
    weather_lst = []
    for reading in data["list"]:
        weather_lst.append(
            [
                reading["dt_txt"],
                reading["main"]["temp"],
                reading["main"]["feels_like"],
                reading["main"]["temp_min"],
                reading["main"]["temp_max"],
                reading["main"]["pressure"],
                reading["main"]["humidity"],
                reading["weather"][0]["main"],
                reading["weather"][0]["description"],
                reading["wind"]["speed"],
                reading["wind"]["gust"],
            ]
        )
    weather_fcast_df = pd.DataFrame(weather_lst, columns=columns)
    return weather_fcast_df


def format_daily_weather_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """Format weather forecast dataframe.

    Args:
        df (pd.DataFrame): Weather forecast dataframe.

    Returns:
        pd.DataFrame: Formatted weather forecast dataframe.
    """
    # select datetime, temp, temp_min, temp_max, windspeed, pressure
    df = df[["datetime", "temp", "wind_speed", "weather_main"]]
    # rename temp to temperature
    df = df.rename(columns={"temp": "temperature"})
    # convert datetime to datetime
    df["datetime"] = pd.to_datetime(df["datetime"])
    # convert datetime to UTC to ESTs
    df = df.assign(datetime=df["datetime"].apply(lambda x: x - timedelta(hours=4)))
    # filter to anytime between 12pm and 8pm
    df = df[(df["datetime"].dt.hour >= 12) & (df["datetime"].dt.hour <= 20)]
    # create flag for clear weather
    df["clear"] = df["weather_main"].apply(lambda x: 1 if x == "Clear" else 0)
    # create flag for cloudy weather
    df["cloudy"] = df["weather_main"].apply(lambda x: 1 if x == "Clouds" else 0)
    # create flag for rain weather
    df["rain"] = df["weather_main"].apply(
        lambda x: 1 if x in ("Rain", "Drizzle", "Thunderstorm") else 0
    )
    # create flag for snow weather
    df["snow"] = df["weather_main"].apply(lambda x: 1 if x == "Snow" else 0)
    # extract date and group by to create daily gameday averages
    df["date"] = df["datetime"].apply(lambda x: x.strftime("%Y-%m-%d"))
    # take the average of the weather conditions
    df = df.groupby("date").mean().reset_index()
    # if rain is > zero, then add flag for rain
    df["is_rain"] = df["rain"].apply(lambda x: 1 if x > 0 else 0)
    # if snow is > zero, then add flag for snow
    df["is_snow"] = df["snow"].apply(lambda x: 1 if x > 0 else 0)
    # drop weather_main, clear, cloudy, rain, snow
    df = df[["date", "temperature", "wind_speed", "is_rain", "is_snow"]]
    return df


if __name__ == "__main__":
    logger.info("collecting weather data")
    args = read_args()
    dir_type, data_type = get_module_purpose(module_path=__file__)
    processed_calendar_path = (
        root_dir
        / "staging_datasets"
        / "season"
        / str(args.season_year)
        / "processed"
        / "calendar"
        / "calendar.csv"
    )
    calendar_df = pd.read_csv(processed_calendar_path)
    calendar_df = filter_calendar_horizon(calendar_df)
    stadium_df = pd.read_csv(root_dir / "staging_datasets" / "stadiums.csv")
    today_dt = datetime.strptime(datetime.now().strftime("%Y-%m-%d"), "%Y-%m-%d")

    game_location_fields = ["date", "team", "opp", "stadium_name", "roof_type"]
    weather_fields = ["temperature", "is_rain", "is_snow", "wind_speed"]
    weather_raw_df = pd.DataFrame(columns=game_location_fields + weather_fields)
    for row in calendar_df.itertuples(index=False):
        game_date, team, opp, is_away = (row.date, row.team, row.opp, row.is_away)
        if is_away == 1:
            home_team = opp
        else:
            home_team = team
        location_dict = stadium_df[stadium_df["team"] == home_team].to_dict(
            orient="records"
        )[0]
        stadium_name, roof_type, lon, lat = (
            location_dict.get("stadium_name"),
            location_dict.get("roof_type"),
            location_dict.get("longitude"),
            location_dict.get("latitude"),
        )
        game_location_data = [game_date, team, opp, stadium_name, roof_type]
        if roof_type not in ["Indoor", "Retractable"]:
            if datetime.strptime(game_date, "%Y-%m-%d") < today_dt:
                weather_df = collect_daily_weather_historicals(
                    game_date=game_date, latitude=lat, longitude=lon
                )
            else:
                weather_df = collect_daily_weather_forecast(
                    latitude=lat, longitude=lon, api_key=args.open_weather_api_key
                )
                weather_df = format_daily_weather_forecast(weather_df)
            weather_df = pd.merge(
                pd.DataFrame(
                    [game_location_data], columns=game_location_fields, index=[0]
                ),
                weather_df,
                how="inner",
                on="date",
            )
        else:
            weather_df = pd.DataFrame(
                [game_location_data], columns=game_location_fields
            )
        weather_raw_df = pd.concat([weather_raw_df, weather_df])

    weather_raw_df.write_ff_csv(
        root_dir=root_dir,
        season_year=args.season_year,
        dir_type=dir_type,
        data_type=data_type,
    )
    logger.info("weather data collected")
