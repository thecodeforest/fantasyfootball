import pandas as pd

from fantasyfootball.config import root_dir, stats_url
from fantasyfootball.pipeline.utils import (
    get_module_purpose,
    read_args,
    write_ff_csv
)


def collect_calendar(calendar_url: str) -> pd.DataFrame:
    calendar_df = pd.read_html(calendar_url)[0]
    return calendar_df


if __name__ == "__main__":
    args = read_args()
    dir_type, data_type = get_module_purpose(module_path=__file__)
    calendar_url = f"{stats_url}/years/{args.season_year}/games.htm"
    calendar_raw = collect_calendar(calendar_url=calendar_url)
    calendar_raw.write_ff_csv(
        root_dir=root_dir,
        season_year=args.season_year,
        dir_type=dir_type,
        data_type=data_type,
    )
