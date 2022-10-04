import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests

sys.path.append(str(Path.cwd()))
from pipeline.pipeline_config import betting_url, root_dir, header  # noqa: E402
from pipeline.utils import (  # noqa: E402
    get_module_purpose,
    retrieve_team_abbreviation,
    fetch_current_week,
    read_args,
    read_ff_csv,
    write_ff_csv,
)


def collect_historical_betting(season_year: int) -> pd.DataFrame:
    url = f"https://www.sportsoddshistory.com/nfl-game-season/?y={season_year}"
    request = requests.get(url, headers=header)
    df = pd.read_html(request.text)[5]
    df.columns = [
        "dow",
        "date",
        "time",
        "home",
        "favored_team",
        "score",
        "spread",
        "place",
        "underdog_team",
        "over_under",
        "placeholder",
    ]
    # drop the last row of data
    df = df.iloc[:-1]
    df["favored_team"] = df["favored_team"].apply(
        lambda x: retrieve_team_abbreviation(x)
    )
    df["underdog_team"] = df["underdog_team"].apply(
        lambda x: retrieve_team_abbreviation(x)
    )
    df["spread"] = df["spread"].apply(lambda x: abs(float(x.split(" ")[1]) / 2))
    df["over_under"] = df["over_under"].apply(lambda x: float(x.split(" ")[1]) / 2)
    df["projected_favorite"] = (df["over_under"] + df["spread"]).round(0)
    df["projected_underdog"] = (df["over_under"] - df["spread"]).round(0)
    # parse the game date 'Sep 8, 2022'
    df["date"] = df["date"].apply(lambda x: datetime.strptime(x, "%b %d, %Y"))
    out_df = pd.DataFrame()
    for row in df.itertuples():
        row1 = [
            row.favored_team,
            row.underdog_team,
            row.projected_favorite,
            row.date,
            2022,
        ]
        row2 = [
            row.underdog_team,
            row.favored_team,
            row.projected_underdog,
            row.date,
            2022,
        ]
        out_df = pd.concat(
            [
                out_df,
                pd.DataFrame(
                    [row1, row2],
                    columns=["team", "opp", "projected_off_pts", "date", "season_year"],
                ),
            ]
        )
    return out_df


def collect_betting(betting_url: str) -> pd.DataFrame:
    request = requests.get(betting_url, headers=header)
    raw_betting_data = pd.DataFrame()
    for matchup in range(0, 16):
        try:
            df = pd.read_html(request.text)[matchup]
            # exclude last column
            df = df[df.columns[:-1].tolist()]
            # rename columns
            df.columns = ["team", "spread", "moneyline", "total_points"]
            raw_betting_data = pd.concat([raw_betting_data, df], axis=0)
        # catch IndexError
        except IndexError:
            # Someone weeks there will be buys;
            # unclear how many games will be played each week
            pass
    return raw_betting_data


if __name__ == "__main__":
    args = read_args()
    dir_type, data_type = get_module_purpose(module_path=__file__)
    data_dir = (
        root_dir / "staging_datasets" / "season" / str(args.season_year) / "processed"
    )
    calendar_df = read_ff_csv(data_dir / "calendar")
    if args.is_historical:
        betting_raw_historical = collect_historical_betting(
            season_year=args.season_year
        )
        betting_raw_historical.write_ff_csv(
            root_dir=root_dir,
            season_year=args.season_year,
            dir_type=dir_type,
            data_type=data_type,
        )
    else:
        betting_raw = collect_betting(betting_url=betting_url)
        current_season_week = fetch_current_week(calendar_df)
        betting_raw.write_ff_csv(
            root_dir,
            args.season_year,
            dir_type,
            data_type,
            "week_" + current_season_week,
        )
