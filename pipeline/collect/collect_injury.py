import sys
from pathlib import Path
import re
import time
import urllib.request
from itertools import chain
from typing import List

import pandas as pd
from bs4 import BeautifulSoup as bs
from bs4.element import Tag

sys.path.append(str(Path.cwd()))
from pipeline_config import injury_url, root_dir  # noqa: E402
from pipeline_logger import logger  # noqa: E402
from utils import get_module_purpose, read_args, write_ff_csv  # noqa: E402

USER_AGENT = "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7"  # noqa: E501
HEADERS = headers = {"User-Agent": USER_AGENT}


def trim_weekly_injury_report_data(team_injury_report: Tag) -> str:
    """Remove the unnecessary data from the weekly injury report
       at the top and bottom of the page.

    Args:
        team_injury_report (Tag): The weekly injury report.

    Returns:
        str: The trimmed weekly injury report.
    """
    team_injury_report = str(team_injury_report)
    report_lines = team_injury_report.split("\n")
    report_lines_index = [
        (index, value) for index, value in enumerate(team_injury_report.split("\n"))
    ]
    start_end_index = [
        (index, value) for index, value in report_lines_index if value == "</div>"
    ]
    start_index, _, end_index, _ = chain(*start_end_index)
    trimmed_injury_report = "\n".join(
        report_lines[start_index + 1 : end_index]  # noqa: E203
    )
    return trimmed_injury_report


def _extract_day_of_week_status(line: str) -> str:
    """Helper function to extract the status for each day of the week
       from the weekly injury report. For example, if the player
       has a Thursday game, status reports will only be
       available for Monday-Wednesday.

    Args:
        line (str): The line from the weekly injury report.

    Returns:
        str: The status for each day of the week.

    """
    day_of_week = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_of_week_regex = " \d\d/\d\d: (.*?)</div>"
    full_week_status = list()
    for day in day_of_week:
        try:
            weekday_regex = day + day_of_week_regex
            daily_status = re.search(re.compile(weekday_regex), line).group(1)
            full_week_status.append(daily_status)
        except Exception:
            daily_status = ""
            full_week_status.append(daily_status)
    return full_week_status


def parse_weekly_injury_data(team_injury_report: str) -> List[str]:
    """Extracts the following information for each player from the weekly injury report:
        * Player name
        * Player position
        * Injury type (if applicable)
        * Status for each day of the week (e.g., DNP, Limited, Full, etc.)
        * Status on the day of the game
       Note that each report is at the team level.

    Args:
        team_injury_report (str): The weekly injury report.

    Returns:
        List[str]: The parsed weekly injury report data.
    """
    parsed_injury_report = list()
    for line in team_injury_report.split("td w15 hidden-xs"):
        if len(line.split("\n")) < 4:
            continue
        player_name = re.search(r"title=\"(.*?) Stats", line).group(1)
        player_position = re.search(
            r"</a></b>, (.*?)<span class=\"visible-xs-inline\">", line
        ).group(1)
        injury_type = re.search(
            r"class=\"visible-xs-inline\"> \((.*?)\)<br/>", line
        ).group(1)
        mon, tue, wed, thu, fri, sat, sun = _extract_day_of_week_status(line)
        game_status = re.search(r"Status:(.*?)</div>", line).group(1)
        parsed_injury_report.append(
            [
                player_name,
                player_position,
                injury_type,
                mon,
                tue,
                wed,
                thu,
                fri,
                sat,
                sun,
                game_status,
            ]
        )
    return parsed_injury_report


if __name__ == "__main__":
    args = read_args()
    dir_type, data_type = get_module_purpose(module_path=__file__)
    injury_raw_df = pd.DataFrame()
    if args.season_year < 2016:
        raise ValueError("Injury data is only available for 2016 and later seasons.")
    logger.info(f"collecting injury data for {args.season_year}")
    for week in range(1, 18):
        weekly_injury_url = f"{injury_url}?yr={args.season_year}&wk={week}&type=reg"
        request = urllib.request.Request(weekly_injury_url, None, HEADERS)
        response = urllib.request.urlopen(request)
        injury_data_soup = bs(response.read(), "html.parser")
        # extract all teams from the injury report
        all_teams = injury_data_soup.findAll("div", {"class": "teamsectlabel"})
        all_teams = [
            re.search(r"<div class=\"teamsectlabel\"><b>(.*?)</b></div>", str(x)).group(
                1
            )
            for x in all_teams
        ]
        # extract all reports for each team
        all_reports = injury_data_soup.findAll(
            "div", {"class": "divtable divtable-striped divtable-mobile"}
        )
        # collect each of the reports
        all_weekly_injury_reports = list()
        # iterate through each team's report and extract injury data
        for team, report in zip(all_teams, all_reports):
            trimmed_injury_report = trim_weekly_injury_report_data(report)
            parsed_injury_report = parse_weekly_injury_data(trimmed_injury_report)
            parsed_injury_report = [x + [team] for x in parsed_injury_report]
            all_weekly_injury_reports.append(parsed_injury_report)
        # flatten out so each list is report for single player
        all_weekly_injury_reports = list(chain(*all_weekly_injury_reports))
        weekly_injury_report_df = pd.DataFrame(
            all_weekly_injury_reports,
            columns=[
                "name",
                "position",
                "injury_type",
                "mon_status",
                "tue_status",
                "wed_status",
                "thu_status",
                "fri_status",
                "sat_status",
                "sun_status",
                "game_status",
                "team",
            ],
        )
        logger.info(
            f"Collected {weekly_injury_report_df.shape[0]} injuries for week {week}"
        )
        weekly_injury_report_df["week"] = week
        weekly_injury_report_df["season_year"] = args.season_year
        injury_raw_df = pd.concat([injury_raw_df, weekly_injury_report_df])
        time.sleep(5)
    injury_raw_df.write_ff_csv(root_dir, args.season_year, dir_type, data_type)
