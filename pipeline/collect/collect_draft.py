import sys
from pathlib import Path
import requests

import pandas as pd

sys.path.append(str(Path.cwd()))
from pipeline.pipeline_config import root_dir, draft_url  # noqa: E402
from pipeline.pipeline_logger import logger  # noqa: E402
from pipeline.utils import get_module_purpose, read_args, write_ff_csv  # noqa: E402


def collect_average_draft_position(draft_url: str) -> pd.DataFrame:
    """Collects the average draft position for each player by year.

    Args:
        draft_url (str): The url to scrape the draft data from.

    Returns:
        pd.DataFrame: The raw draft data for a single season.

    """
    header = {
        "User-Agent": "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7",  # noqa: E501
        "X-Requested-With": "XMLHttpRequest",
    }
    request = requests.get(draft_url, headers=header)
    draft_df = pd.read_html(request.text)[0]
    draft_df = draft_df.drop(columns=["#", "Unnamed: 10"])
    return draft_df


if __name__ == "__main__":
    args = read_args()
    dir_type, data_type = get_module_purpose(module_path=__file__)
    draft_url = f"{draft_url}{args.season_year}"
    logger.info("Collecting average draft position")
    draft_raw = collect_average_draft_position(draft_url=draft_url)
    draft_raw.write_ff_csv(
        root_dir=root_dir,
        season_year=args.season_year,
        dir_type=dir_type,
        data_type=data_type,
    )
    logger.info("Finished collecting average draft position")
