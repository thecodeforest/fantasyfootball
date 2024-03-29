from pathlib import Path

root_dir = Path(__file__).parent.parent

# betting_url = (
#     "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nfl/nfl%20odds%2020"
# )
header = {
    "User-Agent": "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7",  # noqa: E501
    "X-Requested-With": "XMLHttpRequest",
}
stats_url = "https://www.pro-football-reference.com"
betting_url = "https://sportsdata.usatoday.com/football/nfl/odds"
injury_url = "https://www.footballdb.com/transactions/injuries.html"
draft_url = "https://fantasyfootballcalculator.com/adp/standard/12-team/all/"
data_sources = {
    "calendar": {
        "keys": ["team", "season_year"],
        "cols": ["date", "week", "team", "opp", "is_away", "season_year"],
        "is_required": True,
        "is_forward_looking": False,
    },
    "players": {
        "keys": ["team", "season_year"],
        "cols": ["name", "team", "position", "season_year"],
        "is_required": True,
        "is_forward_looking": False,
    },
    "stats": {
        "keys": ["date", "name", "team", "opp", "is_away"],
        "cols": [
            "pid",
            "name",
            "team",
            "opp",
            "is_active",
            "date",
            "result",
            "is_away",
            "is_start",
            "g_nbr",
            "receiving_rec",
            "receiving_yds",
            "receiving_td",
            "rushing_yds",
            "rushing_td",
            "passing_cmp",
            "passing_yds",
            "passing_td",
            "fumbles_fmb",
            "passing_int",
            "scoring_2pm",
            "punt_returns_td",
        ],
        "is_required": False,
        "is_forward_looking": False,
    },
    "salary": {
        "keys": ["name", "position", "team", "opp", "season_year", "week"],
        "cols": [
            "name",
            "position",
            "season_year",
            "week",
            "fanduel_salary",
        ],
        "is_required": False,
        "is_forward_looking": True,
    },
    "defense": {
        "keys": ["week", "opp", "season_year"],
        "cols": [
            "week",
            "opp",
            "rushing_def_rank",
            "receiving_def_rank",
            "passing_def_rank",
            "season_year",
        ],
        "is_required": False,
        "is_forward_looking": True,
    },
    "weather": {
        "keys": ["date", "team", "opp"],
        "cols": [
            "date",
            "team",
            "opp",
            "stadium_name",
            "is_outdoor",
            "avg_windspeed",
            "max_snow_depth",
            "total_precip",
            "avg_temp",
        ],
        "is_required": False,
        "is_forward_looking": True,
    },
    "injury": {
        "keys": ["name", "team", "position", "week", "season_year"],
        "cols": [
            "name",
            "team",
            "position",
            "season_year",
            "week",
            "injury_type",
            "has_dnp_tag",
            "has_limited_tag",
            "most_recent_injury_status",
            "n_injuries",
        ],
        "is_required": False,
        "is_forward_looking": True,
    },
}

scoring = {
    "draft kings": {
        "scoring_columns": {
            "passing_td": 4,
            "passing_yds": 0.04,
            "passing_int": -1,
            "rushing_td": 6,
            "rushing_yds": 0.1,
            "receiving_rec": 1,
            "receiving_td": 6,
            "receiving_yds": 0.1,
            "fumbles_fmb": -1,
            "scoring_2pm": 2,
            "punt_returns_td": 6,
        },
        "multiplier": {
            "rushing_yds": {"threshold": 100, "points": 3},
            "passing_yds": {"threshold": 300, "points": 3},
            "receiving_yds": {"threshold": 100, "points": 3},
        },
    },
    "yahoo": {
        "scoring_columns": {
            "passing_td": 4,
            "passing_yds": 0.04,
            "passing_int": -1,
            "rushing_td": 6,
            "rushing_yds": 0.1,
            "receiving_rec": 0.5,
            "receiving_td": 6,
            "receiving_yds": 0.1,
            "fumbles_fmb": -2,
            "scoring_2pm": 2,
            "punt_returns_td": 6,
        },
        "multiplier": None,
    },
    "custom": {
        "scoring_columns": {
            "passing_td": 5,
            "passing_yds": 0.04,
            "passing_int": -2,
            "rushing_td": 6,
            "rushing_yds": 0.1,
            "receiving_rec": 0.5,
            "receiving_td": 5,
            "receiving_yds": 0.1,
            "fumbles_fmb": -2,
            "scoring_2pm": 4,
            "punt_returns_td": 6,
        },
        "multiplier": None,
    },
}
