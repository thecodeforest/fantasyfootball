import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path.cwd()))
from pipeline.pipeline_config import root_dir, stats_url  # noqa: E402

from pipeline.utils import get_module_purpose, read_args, write_ff_csv  # noqa: E402


def collect_calendar(calendar_url: str) -> pd.DataFrame:
    """Collects the calendar data for a complete season. Includes
       data on date, time, winner, loser, box-score, total points
       for each team.

    Args:
        calendar_url (str): The url to scrape the calendar data from.

    Returns:
        pd.DataFrame: The calendar data.
    """
    calendar_df = pd.read_html(calendar_url)[0]
    return calendar_df


if __name__ == "__main__":
    args = read_args()
    dir_type, data_type = get_module_purpose(module_path=__file__)
    calendar_url = f"{stats_url}/years/{args.season_year}/games.htm"
    calendar_raw = collect_calendar(calendar_url=calendar_url)
    print(root_dir)
    calendar_raw.write_ff_csv(
        root_dir=root_dir,
        season_year=args.season_year,
        dir_type=dir_type,
        data_type=data_type,
    )
