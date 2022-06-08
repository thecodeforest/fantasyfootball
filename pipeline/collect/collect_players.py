import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path.cwd()))
from pipeline.pipeline_config import root_dir, stats_url  # noqa: E402
from pipeline.utils import get_module_purpose, read_args, write_ff_csv  # noqa: E402


def collect_players(url: str, season_year: int) -> pd.DataFrame:
    """Collects player data, including first name, last name, team,
       and position for a given season.

    Args:
        url (str): The url prefix for all active players in a given season.
        season_year (int): The year of the season.

    Returns:
        pd.DataFrame: The player data for a given season.
    """
    season_year_url = f"{url}/years/{season_year}/fantasy.htm"
    players = pd.read_html(season_year_url)[0]
    players.columns = ["_".join(x) for x in players.columns.to_flat_index()]
    players = players[
        [
            "Unnamed: 1_level_0_Player",
            "Unnamed: 2_level_0_Tm",
            "Unnamed: 3_level_0_FantPos",
        ]
    ]
    players.columns = ["player", "team", "position"]
    players["season_year"] = season_year
    return players


if __name__ == "__main__":
    args = read_args()
    dir_type, data_type = get_module_purpose(module_path=__file__)
    players_raw = collect_players(url=stats_url, season_year=args.season_year)
    players_raw.write_ff_csv(
        root_dir=root_dir,
        season_year=args.season_year,
        dir_type=dir_type,
        data_type=data_type,
    )
