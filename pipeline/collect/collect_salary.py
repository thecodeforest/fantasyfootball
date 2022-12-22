import sys
from pathlib import Path
import requests

import pandas as pd

sys.path.append(str(Path.cwd()))
from pipeline.pipeline_config import root_dir, header  # noqa: E402
from pipeline.utils import (  # noqa: E402
    get_module_purpose,
    fetch_current_week,
    read_args,
    write_ff_csv,
    read_ff_csv,
)  # noqa: E402

"""Collect data based on FanDuel salaries. Data is collected for
the upcoming, unplayed week. This data is then processed in the
process_salary module and appended to the existing salary data,
which is stored in S3. An append will only occur if the data
coming from collect_salary module has a week number that is
greater than the max week in the existing salary data. Otherwise,
an error will be raised.
The URL below is the source of historical Salary data.
url: https://dailyroto.com/nfl-historical-production\
        -fantasy-points-draftkings-fanduel
"""


def collect_salary():
    fanduel_url = "https://www.footballdiehards.com/fantasyfootball/dailygames/FanDuel-Salary-data.cfm"  # noqa: E501
    request = requests.get(fanduel_url, headers=header)
    fanduel_salary_df = pd.read_html(request.text)[0]
    fanduel_salary_df = fanduel_salary_df.droplevel(0, axis=1)
    return fanduel_salary_df


if __name__ == "__main__":
    args = read_args()
    dir_type, data_type = get_module_purpose(module_path=__file__)
    data_dir = (
        root_dir / "staging_datasets" / "season" / str(args.season_year) / "processed"
    )
    calendar_df = read_ff_csv(data_dir / "calendar")
    salary_raw = collect_salary()
    season_week = str(salary_raw["week"].unique()[0])
    current_season_week = fetch_current_week(calendar_df)
    # ensure salary website has the correct, current week
    assert_msg = (
        f"Salary website has week: {season_week}"
        f"but it should be: {current_season_week}"
    )
    assert season_week == current_season_week, assert_msg
    salary_raw.write_ff_csv(
        root_dir, args.season_year, dir_type, data_type, "week_" + current_season_week
    )
