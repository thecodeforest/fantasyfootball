from datetime import datetime
from time import sleep

import pandas as pd
from meteostat import Daily, Point

from fantasyfootball.config import root_dir
from fantasyfootball.pipeline.pipeline_logger import logger
from fantasyfootball.pipeline.utils import get_module_purpose, read_args, write_ff_csv


def collect_daily_weather_conditions(
    game_date: str, latitude: float, longitude: float
) -> pd.DataFrame:
    game_date = datetime.strptime(game_date, "%Y-%m-%d")
    point = Point(lat=latitude, lon=longitude)
    game_day_weather_df = (
        Daily(point, start=game_date, end=game_date).fetch().reset_index()
    )
    return game_day_weather_df


def collect_weather(
    calendar_df: pd.DataFrame, stadium_df: pd.DataFrame
) -> pd.DataFrame:
    game_location_fields = ["date", "team", "opp", "stadium_name", "roof_type"]
    game_weather_df = pd.DataFrame(columns=game_location_fields)
    for row in calendar_df.itertuples(index=False):
        logger.info(f"collecting data for {row}")
        game_date, team, opp, is_away = (row.date, row.team, row.opp, row.is_away)
        if is_away == 1:
            home_team = opp
        else:
            home_team = team
        game_location = stadium_df[stadium_df["team"] == home_team]
        if game_location.empty:
            raise Exception(f"No stadium found for {home_team}")
        stadium_name, roof_type, lon, lat = (
            game_location[["stadium_name", "roof_type", "longitude", "latitude"]]
            .iloc[0]
            .tolist()
        )
        game_location_data = [row.date, row.team, row.opp, stadium_name, roof_type]
        if roof_type in ["Indoor", "Retractable"]:
            game_day_weather_df = pd.DataFrame(
                [game_location_data], columns=game_location_fields
            )
        else:
            game_day_weather_df = collect_daily_weather_conditions(
                game_date=game_date, latitude=lat, longitude=lon
            )
            game_day_weather_df = game_day_weather_df.drop(columns="time")
            game_day_weather_df = pd.DataFrame(
                [game_location_data + game_day_weather_df.iloc[0].tolist()],
                columns=game_location_fields + game_day_weather_df.columns.tolist(),
            )
        game_weather_df = pd.concat([game_weather_df, game_day_weather_df])
        sleep(2)
    return game_weather_df


if __name__ == "__main__":
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
    stadium_df = pd.read_csv(root_dir / "staging_datasets" / "stadiums.csv")
    weather_raw = collect_weather(calendar_df, stadium_df)
    weather_raw.write_ff_csv(
        root_dir=root_dir,
        season_year=args.season_year,
        dir_type=dir_type,
        data_type=data_type,
    )
