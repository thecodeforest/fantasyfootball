import pandas as pd

from fantasyfootball.config import betting_url, root_dir
from fantasyfootball.pipeline.utils import get_module_purpose, read_args, write_ff_csv


def collect_betting(betting_url: str, season_year: int) -> pd.DataFrame:
    """Collects betting data, including team, opponent, over/under, opening spread,
       closing spread, and outcome for each game played so far in a season.
    Args:
        betting_url (str): The url prefix for all betting data in a given season.
        season_year (int): The year of the season.

    Returns:
        pd.DataFrame: The betting data for a given season.
    """
    season_year_abbr = str(season_year)[2:]
    end_of_season_year_abbr = str(season_year + 1)[2:]
    betting_url = f"{betting_url}{season_year_abbr}-{end_of_season_year_abbr}.xlsx"
    betting_df = pd.read_excel(betting_url)
    return betting_df


if __name__ == "__main__":
    args = read_args()
    dir_type, data_type = get_module_purpose(module_path=__file__)
    betting_raw = collect_betting(betting_url=betting_url, season_year=args.season_year)
    betting_raw.write_ff_csv(
        root_dir=root_dir,
        season_year=args.season_year,
        dir_type=dir_type,
        data_type=data_type,
    )
