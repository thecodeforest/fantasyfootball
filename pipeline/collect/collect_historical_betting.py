import requests
import pandas as pd
from datetime import datetime
from pipeline.utils import retrieve_team_abbreviation
from pipeline.pipeline_config import root_dir, header


def collect_historical_betting():
    url = "https://www.sportsoddshistory.com/nfl-game-season/?y=2022"
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


if __name__ == "__main__":
    df = collect_historical_betting()
    df.to_csv(root_dir / "data" / "betting" / "historical_betting.csv", index=False)
