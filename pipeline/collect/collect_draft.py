import pandas as pd
from pathlib import Path
import sys

from bs4 import BeautifulSoup as bs
from bs4.element import Tag
import urllib.request

sys.path.append(str(Path.cwd()))
from pipeline.pipeline_config import root_dir, draft_url, header  # noqa: E402
from pipeline.pipeline_logger import logger  # noqa: E402
from pipeline.utils import get_module_purpose, read_args, write_ff_csv  # noqa: E402


def collect_draft() -> bs:
    url = "https://www.fantasypros.com/nfl/adp/overall.php"
    request = urllib.request.Request(url, None, header)
    response = urllib.request.urlopen(request)
    draft_data_soup = bs(response.read(), "html.parser")
    return draft_data_soup


def prep_raw_draft(draft_data_soup: bs) -> pd.DataFrame:
    player_labels = draft_data_soup.find_all("td", class_="player-label")
    player_draft_info = []
    for index, player_label in enumerate(player_labels):
        player_name = player_label.find("a").text
        try:
            player_team = player_label.find("small").text
        # handle empty values
        except Exception as e:
            print(e)
            player_team = "FA"
        player_draft_position = index
        player_draft_info.append([player_name, player_team, player_draft_position])
    raw_draft_df = pd.DataFrame(
        player_draft_info, columns=["name", "team", "avg_draft_position"]
    )
    return raw_draft_df


if __name__ == "__main__":
    args = read_args()
    dir_type, data_type = get_module_purpose(module_path=__file__)
    draft_url = f"{draft_url}{args.season_year}"
    logger.info("Collecting draft position")
    draft_raw_bs = collect_draft()
    draft_raw_df = prep_raw_draft(draft_raw_bs)
    draft_raw_df.write_ff_csv(
        root_dir=root_dir,
        season_year=args.season_year,
        dir_type=dir_type,
        data_type=data_type,
    )
    logger.info("Finished collecting draft position")
