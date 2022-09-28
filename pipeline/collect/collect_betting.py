import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path.cwd()))
from pipeline.pipeline_config import betting_url, root_dir, header  # noqa: E402
from pipeline.utils import (  # noqa: E402
    get_module_purpose,
    fetch_current_week,
    read_args,
    read_ff_csv,
    write_ff_csv,
)


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
    betting_raw = collect_betting(betting_url=betting_url)
    current_season_week = fetch_current_week(calendar_df)
    betting_raw.write_ff_csv(
        root_dir, args.season_year, dir_type, data_type, "week_" + current_season_week
    )
