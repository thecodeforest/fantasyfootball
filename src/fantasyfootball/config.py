from pathlib import Path

root_dir = Path(__file__).parent
stats_url = "https://www.pro-football-reference.com"
betting_url = (
    "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nfl/nfl%20odds%2020"
)
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
    "betting": {
        "keys": ["date", "season_year", "team", "opp"],
        "cols": ["team", "opp", "projected_off_pts", "date", "season_year"],
        "is_required": False,
        "is_forward_looking": True,
    },
    "salary": {
        "keys": ["name", "position", "team", "opp", "season_year", "week"],
        "cols": [
            "name",
            "position",
            "week",
            "team",
            "opp",
            "fanduel_salary",
            "draftkings_salary",
            "season_year",
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
}
