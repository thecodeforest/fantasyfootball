import sys
from pathlib import Path
import re

import time
from itertools import chain, product
from typing import List, Tuple
from datetime import datetime, timezone

import pandas as pd
import requests
import boto3

sys.path.append(str(Path.cwd()))
from pipeline.pipeline_config import root_dir, stats_url  # noqa: E402
from pipeline.pipeline_logger import logger  # noqa: E402
from pipeline.utils import get_module_purpose, read_args, write_ff_csv  # noqa: E402


def create_url_by_season(
    url: str, last_name: str, player_id: str, season_year: int
) -> str:
    """Creates the url to scrape player stats by season.

    Args:
        player_name_first_letter (str): player's name.
        player_id (str): The player's id.
        season_year (int): The season year to scrape player stats from.

    Returns:
        str: The url to scrape player stats by season.
    """
    last_name_first_letter = last_name[0]
    url = f"{url}/players/{last_name_first_letter}/{player_id}/gamelog/{season_year}/"
    return url


def first_name_is_abbr(first_name: str) -> bool:
    """Checks if first name is an abbreviation (e.g. 'J.R.', 'D.K.').
       Some player IDs are not created correctly when first name is an abbreviation,
       as there are several ways a player's ID can be created with an abbreviation.

    Args:
        first_name (str): The first name of the player.

    Returns:
        bool: True if first name is an abbreviation, False otherwise.
    """
    pattern = re.compile(r"[A-Z]\.[A-Z]\.")
    is_abbr = re.match(pattern, first_name)
    if is_abbr:
        return True
    else:
        return False


def clean_player_name(name: str, name_part: str) -> str:
    """Removes punctuation from player's name. Periods and
       dashes are removed from both first and last names, while
       apostrophes are only removed from last names.

    Args:
        name (str): The name of the player.
        name_part (str): The part of the name to clean (first or last).

    Returns:
        str: The cleaned name.
    """
    if name_part not in ["first", "last"]:
        raise ValueError(f"{name_part} is not a valid. Must 'first' or 'last'.")
    if name_part == "last":
        chars_to_remove_regex = r"\.|\-"
    if name_part == "first":
        chars_to_remove_regex = r"\.|\-|\'"
    name = re.sub(chars_to_remove_regex, "", name)
    return name


def create_abbr_name_combo(first_name: str) -> Tuple[str, str, str]:
    """Creates a tuple of all (known) possible combinations of an
       abbreviated first name.

    Args:
        first_name (str): The first name of the player.

    Returns:
        Tuple[str, str, str]: A tuple of all (known) possible combinations
        of an abbreviated first name.For example, a player with the first name
        of D.J. and last name of Smith can have an ID of
        'SmitDJ00', SmitD.00', 'SmitD00'.

    """
    first_name_full = first_name.replace(".", "")[:2]
    first_name_single_period = first_name[:2]
    first_name_single_letter = first_name[:1]
    return (first_name_full, first_name_single_period, first_name_single_letter)


def create_apostrophe_name_combo(last_name: str) -> Tuple[str, str]:
    """Creates a tuple of all (known) possible combinations of a
       last name with an apostrophe. For example, using the first
       four characters of the last name to create a player id for
       'James O'Shaughnessys' would return ('O'Sh', 'OSha').

    Args:
        last_name (str): The last name of the player.

    Returns:
        Tuple[str, str]: A tuple of all (known) possible combinations
    """
    last_name_no_apostrophe = last_name.replace("'", "")[:4]
    last_name_apostrophe = last_name[:4]
    return (last_name_no_apostrophe, last_name_apostrophe)


def create_player_id(first_name: str, last_name: str) -> List[str]:
    """Creates a list of all possible player ids for a given player. A player id (pid)
       is required to complete the url that references their statistics. A Pid
       typically contains the two letters of a player's first name (Aa),
       the first four letters (Rodg) of their last name, and a zero padded number (00)
       to create a unique identifer for a player (Aaron Rodgers = RodgAa00).
       Multiple pids are generated for each player
       for several reasons, including:
       * Players have similiar names. For example, Demone Harris is 'HarrDe06' and
         Deonte Harris is 'HarrDe07'.
       * Inconsistent naming conventions for players with abbreviated first names.
         Abbreviated first names can contain no periods (DJ), a single period (D.),
         or just the the first letter (D).
       * Inconsistent naming conventions for players with apostrophes in
         their last name. For example, 'James O'Shaughnessy' has a pid
         of 'O'ShJa00', whereas 'Ken O'Brian' has a pid of 'OBriKe00'.


    Args:
        first_name (str): The first name of the player.
        last_name (str): The last name of the player.

    Returns:
        List[str]: A list of all possible player ids for a given player.
    """
    player_index = ["0" + str(x) if x <= 9 else str(x) for x in range(0, 10)]
    if first_name_is_abbr(first_name):
        (
            first_name_full,
            first_name_single_period,
            first_name_single_letter,
        ) = create_abbr_name_combo(first_name)
        playerids = zip(
            product([first_name_full], player_index),
            product([first_name_single_period], player_index),
            product([first_name_single_letter], player_index),
        )
        playerids = ["".join(x) for x in chain(*playerids)]
        playerids = [last_name[:4] + x for x in playerids]
    if "'" in last_name and not first_name_is_abbr(first_name):
        last_name_no_apostrophe, last_name_apostrophe = create_apostrophe_name_combo(
            last_name
        )
        playerids = zip(
            product([last_name_apostrophe + first_name[:2]], player_index),
            product([last_name_no_apostrophe + first_name[:2]], player_index),
        )
        playerids = ["".join(x) for x in chain(*playerids)]
    if "'" not in last_name and not first_name_is_abbr(first_name):
        first_name = clean_player_name(name=first_name, name_part="first")
        player_id = "".join(last_name)[:4] + first_name[:2]
        playerids = [
            pt1 + pt2 for pt1, pt2 in zip([player_id] * len(player_index), player_index)
        ]
    return playerids


def pad_last_name(last_name: str) -> str:
    """Pad the last name with 'x' to make it a valid player id.
       For example, 'Jack Ham' becomes 'HamxJa00'.

    Args:
        last_name (str): The last name of the player.

    Returns:
        str: The last name of the player with 'x' padded.
    """
    padding = "".join((4 - len(last_name)) * ["x"])
    last_name += padding
    return last_name


# TO DO: Add a function to check which players already exist in S3
def get_recent_raw_stats(
    season_year: int, bucket_name: str = "fantasy-football-pipeline"
) -> set:
    """Get the contents of the S3 bucket that holds the raw player
       stats for a given season year.If the bucket is empty,
       return an empty set. If the bucket is not empty and the
       contents are from today, return the contents.

    Args:
        season_year (int): The season year for which to get the
            contents of the S3 bucket.
        bucket_name (str, optional): The name of the S3 bucket.
            Defaults to "fantasy-football-pipeline".

    Returns:
        set: A set of the contents of the S3 bucket for the
            given season year. (e.g., {'MinsGa00', ''HurtJa00'})

    """
    key = f"datasets/season/{season_year}/raw/stats"
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)
    # if bucket is empty, return empty dict
    if not list(bucket.objects.filter(Prefix=key)):
        return {}
    today_date = datetime.now().astimezone(timezone.utc).strftime("%Y-%m-%d")
    bucket_contents = set()
    for obj in bucket.objects.filter(Prefix=key):
        # get the last modified date of the object
        last_modified_date = obj.last_modified.strftime("%Y-%m-%d")
        if last_modified_date != today_date:
            continue
        # get the name of the object - e.g.,
        # 'datasets/season/2022/raw/stats/MinsGa00_stats.csv'
        player_id = obj.key.split("/")[-1].replace("_stats.csv", "")
        bucket_contents.add(player_id)
    return bucket_contents


def collect_stats(
    player_name: str,
    player_team: str,
    season_year: int,
    stats_url: str,
    existing_player_data: set,
    header: dict,
) -> pd.DataFrame:
    """Collects the season stats for a given player.

    Args:
        player_name (str): The name of the player.
        player_team (str): The 3 letter abbreviation of the player's team.
        season_year (int): The year of the season.
        stats_url (str): The url prefix for the player's stats.

    Returns:
        pd.DataFrame: The season stats for a given player.
    """
    player_id_edge_cases = {
        "Derek Carr": "CarrDe02",
        "Derek Carrier": "CarrDe00",
        "Equanimeous St. Brown": "St.BEq00",
        "Deonte Harty": "HarrDe07",
    }
    first_name, *last_name = player_name.split(" ")
    last_name = last_name[0]
    if len(last_name) < 4:
        last_name = pad_last_name(last_name)
    last_name = clean_player_name(name=last_name, name_part="last")
    player_ids = create_player_id(first_name=first_name, last_name=last_name)
    for player_id in player_ids:
        if player_id in existing_player_data:
            logger.info(f"Player {player_id} already exists in S3. Skipping...")
            return pd.DataFrame()
    for player_id in player_ids:
        if player_name in player_id_edge_cases.keys():
            player_id = player_id_edge_cases.get(player_name)
        player_url = create_url_by_season(stats_url, last_name, player_id, season_year)
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36",  # noqa E501
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "TE": "Trailers",
            }
            stats = pd.read_html(requests.get(player_url, headers=headers).content)[0]
            # stats = pd.read_html(player_url, header=header)[0]
            stats.columns = ["_".join(x) for x in stats.columns.to_flat_index()]
            # Ensures player is on correct team
            correct_team = (
                player_team in stats["Unnamed: 5_level_0_Tm"].dropna().unique()
            )
            # Ensure player doesn't already exist in staging data
            # (e.g., Derek Carrier & Derek Carr)
            raw_stats_csvs = (
                root_dir
                / "staging_datasets"
                / "season"
                / str(season_year)
                / "raw"
                / "stats"
            ).glob("*.csv")
            stats_do_not_exist = player_id not in [
                x.stem.replace("_stats", "") for x in raw_stats_csvs
            ]
            # If both conditions are met, add player to staging data
            if correct_team and stats_do_not_exist:
                stats["pid"] = player_id
                stats["name"] = player_name
                return stats
        except Exception as e:
            logger.error(f"Error collecting stats for {player_url}")
            logger.error(e)
            if e == "HTTP Error 429: Too Many Requests":
                # shut down the script
                logger.error("Too many requests. Shutting down...")
                sys.exit()
                # logger.error("Sleeping for 5 minutes...")
                # time.sleep(300)
                # logger.error("Resuming...
            continue
    return pd.DataFrame(None)


if __name__ == "__main__":
    args = read_args()
    dir_type, data_type = get_module_purpose(module_path=__file__)
    data_dir = (
        root_dir
        / "staging_datasets"
        / "season"
        / str(args.season_year)
        / dir_type
        / data_type
    )
    players = pd.read_csv(
        data_dir.parent.parent / "processed" / "players" / "players.csv"
    )
    existing_player_data = get_recent_raw_stats(season_year=args.season_year)
    for row in players.itertuples():
        _, player_name, player_team, player_position, _ = row
        logger.info(f"collecting data for {player_name}")
        stats_raw = collect_stats(
            player_name=player_name,
            player_team=player_team,
            season_year=args.season_year,
            stats_url=stats_url,
            existing_player_data=existing_player_data,
        )
        if stats_raw.empty:
            logger.error(f"Could not collect stats for {player_name}")
            continue
        pid = stats_raw["pid"].iloc[0]
        stats_raw.write_ff_csv(root_dir, args.season_year, dir_type, data_type, pid)
        time.sleep(5)
