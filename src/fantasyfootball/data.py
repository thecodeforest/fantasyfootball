from __future__ import annotations  # noqa: F404

import logging
from pathlib import PosixPath
from typing import List
from urllib.error import HTTPError

import numpy as np
import pandas as pd
from janitor import clean_names

from fantasyfootball.config import data_sources, root_dir, scoring

logger = logging.getLogger("fantasydata")
logger.setLevel(logging.INFO)


class FantasyData:
    """Loads historical fantasy football data.

    Args:
        season_year_start (int): The first year of the season.
        season_year_end (int): The last year of the season.
    """

    def __init__(self, season_year_start: int, season_year_end: int):
        self.season_year_start = season_year_start
        self.season_year_end = season_year_end
        self._validate_season_year_range()
        # Set when FantasyData object is created.
        self.ff_data = None
        self.load_data()
        self.scoring = scoring

    def _validate_season_year_range(self) -> bool:
        """Ensures that the season year range is valid.

        Raises:
            ValueError: If the season year is less than the minimum year
            ValueError: If the season year is greater than the maximum year

        Returns:
            bool: True if the season year range is valid.
        """
        season_years = [
            int(str(x).split("/")[-1])
            for x in (root_dir / "datasets" / "season").glob("*")
            if str(x).split("/")[-1].isdigit()
        ]

        min_year = min(season_years)
        max_year = max(season_years)
        if self.season_year_start < min_year:
            raise ValueError(
                f"Season year start {self.season_year_start}\n"
                f"is less than minimum year {min_year}"
            )
        if self.season_year_end > max_year:
            raise ValueError(
                f"Season year end {self.season_year_end}\n"
                f" is greater than maximum year {max_year}"
            )
        return True

    @staticmethod
    def _refresh_data(ff_data_dir: PosixPath, data_sources: dict) -> bool:
        """Use the datasets specified in the `config.py` to identify
        if a dataset is missing from the installed version of the package.
        When a missing dataset is identified, the most recent version
        is downloaded from Git.

        Args:
            ff_data_dir (PosixPath): The directory containing the seasonal data.
            data_sources (dict): A dictionary indicating the names of
            the data sources used in the fantasyfootball package.

        Returns:
            bool: True if data in current package is up to date or
            data was succesfully downloaded from remote repo.
        """
        season_year = ff_data_dir.name
        base_url = f"https://github.com/thecodeforest/fantasyfootball/blob/main/datasets/season/{season_year}"  # noqa E501
        expected_data_sources = data_sources.keys()
        local_data_sources = [
            x.name.replace(".gz", "") for x in ff_data_dir.glob("*.gz")
        ]
        missing_local_data_sources = set(expected_data_sources) - set(
            local_data_sources
        )
        if missing_local_data_sources:
            for missing_data in missing_local_data_sources:
                missing_data_url = f"{base_url}/{missing_data}.gz?raw=true"
                try:
                    logger.info(f"Fetching most recent {missing_data} data from remote")
                    missing_data_df = pd.read_csv(missing_data_url, compression="gzip")
                    missing_data_df.to_csv(
                        ff_data_dir / f"{missing_data}.gz",
                        index=False,
                        compression="gzip",
                    )
                    logger.info(
                        f"Succussfully downloaded {missing_data} data from remote"
                    )
                except HTTPError as error:
                    logger.error("Error:", error)
                    logger.error(f"{missing_data} data not available on remote")
                    raise
        return True

    @staticmethod
    def _load_data(
        ff_data_dir: PosixPath, data_sources: dict, *exclude: str
    ) -> pd.DataFrame:
        """Helper method to load all other data, excluding the
        season calendar and roster of active players for a season.

        Args:
            ff_data_dir (PosixPath): The directory containing the season data.
            data_sources (dict): A dictionary indicating the names of
                the data sources used in the fantasyfootball package.
            exclude (str): The names of the files to exclude from the data load.

        Raises:
            ValueError: If the exclude file name is a required file.
            ValueError: If the columns used to join the data are not found.

        Returns:
            pd.DataFrame: The dataframe containing all
            historical fantasy football data for a single season.
        """

        required_data = [
            k for k in data_sources.keys() if data_sources[k]["is_required"]
        ]
        if set(exclude).intersection(required_data):
            raise ValueError(
                f"Cannot exclude required data: {required_data}. "
                f"Please do not exclude required data and try again."
            )
        calendar_df = pd.read_csv(ff_data_dir / "calendar.gz", compression="gzip")
        players_df = pd.read_csv(ff_data_dir / "players.gz", compression="gzip")
        season_ff_df = pd.merge(
            calendar_df, players_df, how="inner", on=["team", "season_year"]
        )
        supplementary_data = set(data_sources.keys()) - set(required_data)
        for data in data_sources:
            if data in exclude:
                continue
            if data in supplementary_data:
                dataset_df = pd.read_csv(ff_data_dir / f"{data}.gz", compression="gzip")
                keys = data_sources[data]["keys"]
                if not set(keys).issubset(set(dataset_df.columns)):
                    raise ValueError(
                        f"{data} does not contain all the required keys: {keys}"
                    )
                season_ff_df = pd.merge(season_ff_df, dataset_df, on=keys, how="left")
        return season_ff_df

    def load_data(
        self, data_sources: dict = data_sources, filter_final_season_week: bool = True
    ) -> FantasyData:
        """Loads all historical fantasy football data from the season year
        range provided. Each season year is loaded separately and
        then concatenated together.

        Args:
            data_sources (dict): A dictionary indicating the names of
                the data sources used in the fantasyfootball package.
            filter_final_season_week (bool): If True, the final week of each
                season is filtered out. These weeks are filtered because many players
                are 'active' but have minimal participation in
                the game to avoid injury when their team has secured a playoff spot.
                Excluding these weeks allows for more accurate predictions.
                Default is True.

        Returns:
            FantasyData: The dataframe containing all historical
                fantasy football data for the specified season year range.
        """
        logger.info(
            f"Loading data from {self.season_year_start} to {self.season_year_end}"
        )
        ff_data_dir = root_dir / "datasets" / "season"
        ff_df = pd.DataFrame()
        for season_year in range(self.season_year_start, self.season_year_end + 1):
            if season_year < 2016:
                logger.warning("Player injury data not available prior to 2016 season")
            # TO DO: Do not throw expection if offline; raise warning instead
            self._refresh_data(ff_data_dir / str(season_year), data_sources)
            season_ff_df = self._load_data(ff_data_dir / str(season_year), data_sources)
            if filter_final_season_week:
                max_week = max(season_ff_df["week"])
                logger.info(
                    f"Dropping final week (week {max_week}) of season {season_year}"
                )
                season_ff_df = season_ff_df[season_ff_df["week"] != max_week]
            ff_df = pd.concat([ff_df, season_ff_df])
        self.ff_data = ff_df
        return FantasyData

    @staticmethod
    def _validate_scoring_source_rules(source_rules: dict, ff_df_columns: list) -> None:
        """Validates the scoring source rules provided.

        Args:
            source_rules (dict): The scoring source rules to validate.
            ff_df_columns (list): The list of columns in the dataframe.

        Raises:
            ValueError: If the scoring source rules are not valid.
            TypeError: If the scoring source rules are not a dictionary.
            KeyError: If the scoring source keys do not include
                'scoring_columns' or 'multiplier'.
            KeyError: If scoring columns are not a subset of the dataframe columns.
            TypeError: If the scoring multiplier is not a dictionary.
            KeyError: If the scoring multiplier keys do not include
                'threshold' and 'points'.
            KeyError: If the multiplier values are not a subset of
                the dataframe columns.
        """
        # validate that source name is present
        source_name = list(source_rules.keys())[0]
        if not source_name:
            raise ValueError("Source name is required")
        scoring_rules = source_rules[source_name]
        if not isinstance(source_rules, dict):
            raise TypeError("scoring_source_rules must be a dictionary")
        # validate required keys for scoring
        required_keys = {"scoring_columns", "multiplier"}
        if not set(scoring_rules.keys()) == required_keys:
            raise KeyError(f"Scoring rules must contain keys {required_keys}")
        # validate scoring columns are subset of columns present in data columns
        if not set(scoring_rules["scoring_columns"].keys()) <= set(ff_df_columns):
            raise KeyError(
                "Scoring columns must be a subset of columns present in dataframe"
            )
        # validate multiplier is a dictionary
        if not isinstance(scoring_rules["multiplier"], dict):
            raise TypeError("scoring_rules['multiplier'] must be a dictionary")
        # validate if multiplier is present, it has to have threhold and points keys
        if scoring_rules["multiplier"] is not None:
            for column_multipler in scoring_rules["multiplier"].keys():
                if not set(scoring_rules["multiplier"][column_multipler].keys()) == {
                    "threshold",
                    "points",
                }:
                    raise KeyError(
                        f"scoring_rules['multiplier'][{column_multipler}] must\n"
                        f"have 'threshold' and 'points' keys"
                    )
        # validate multiplier scoring column is subset of scoring columns
        if scoring_rules["multiplier"] is not None:
            if not set(scoring_rules["multiplier"].keys()) <= set(
                scoring_rules["scoring_columns"].keys()
            ):
                raise KeyError(
                    "Multiplier scoring column must be a subset of scoring columns"
                )

    def add_scoring_source(self, source_rules: dict) -> FantasyData:
        """Updates the scoring source rules.

        Args:
            source_rules (dict): Scoring source rules. Required keys
                are the source name (e.g., 'custom'), 'scoring_columns',
                and 'multiplier'.

        Returns:
            FantasyData: An updated FantasyData object with
                the new scoring source rules.

        Example:
            >>> from fantasyfootball.data import FantasyData
            >>> fantasy_data = FantasyData(season_year_start=2019,
                                           season_year_end=2021
                                           )
            >>> new_scoring_source = {"my league": {"scoring_columns": {
                    "passing_td": 4,
                    "passing_yds": 0.04,
                    "passing_int": -3,
                    "rushing_td": 6,
                    "rushing_yds": 0.1,
                    "receiving_rec": 0.5,
                    "receiving_td": 4,
                    "receiving_yds": 0.1,
                    "fumbles_fmb": -3,
                    "scoring_2pm": 4,
                    "punt_returns_td": 6,
                    },
                    "multiplier": {"rushing_yds" : {"threshold": 100,"points": 5},
                                    "passing_yds": {"threshold": 300, "points": 3},
                                    "receiving_yds": {"threshold": 100, "points": 3},
                    }
                    }}
            >>> fantasy_data.add_scoring_source(new_scoring_source)
        """
        self._validate_scoring_source_rules(source_rules, self.ff_data.columns.tolist())
        source_name = list(source_rules.keys())[0]
        self.scoring[source_name] = source_rules
        logger.info(f"Scoring source '{source_name}' Added")
        return FantasyData

    @staticmethod
    def score_player(
        player_df: pd.DataFrame, scoring_columns: set, scoring_source_rules: dict
    ) -> np.array:
        """Calculates the total number of points scored for a single week

        Args:
            player_df (pd.DataFrame): Weekly stats for a single player for the season.
            scoring_columns (set): Columns to use for scoring
            scoring_source_rules (dict): Rules for scoring

        Returns:
            np.array: The total number of points scored for a single week.
        """
        player_weekly_points = [0] * player_df.shape[0]
        for column in scoring_columns:
            point_amount = scoring_source_rules["scoring_columns"][column]
            scoring_amount = player_df[column]
            weekly_points_scored = [x * point_amount for x in scoring_amount]
            player_weekly_points = np.add(player_weekly_points, weekly_points_scored)
            if scoring_source_rules.get("multiplier"):
                column_multiplier = scoring_source_rules["multiplier"].get(column)
                if column_multiplier:
                    weekly_mult_points_scored = [
                        column_multiplier["points"]
                        if x > column_multiplier["threshold"]
                        else 0
                        for x in scoring_amount
                    ]
                    player_weekly_points = np.add(
                        player_weekly_points, weekly_mult_points_scored
                    )
        return player_weekly_points

    def create_fantasy_points_column(self, scoring_source: str) -> FantasyData:
        """Creates a fantasy points column for the scoring source provided.

        Args:
            scoring_source (str): Name of the scoring source to use
                (e.g., 'draft kings', 'yahoo', 'custom').

        Returns:
            FantasyData: An updated FantasyData object with the new
                fantasy points column.
        """
        scoring_source_rules = self.scoring[scoring_source]
        scoring_columns = set(scoring_source_rules["scoring_columns"].keys()) & set(
            self.ff_data.columns
        )
        all_player_pts = list()
        for row in self.ff_data[["name", "pid"]].drop_duplicates().itertuples():
            player_df = self.ff_data[
                (self.ff_data["name"] == row.name) & (self.ff_data["pid"] == row.pid)
            ]
            player_weekly_points = self.score_player(
                player_df, scoring_columns, scoring_source_rules
            )
            all_player_pts.append(
                [row.name, row.pid, player_df["date"], player_weekly_points]
            )
        all_pts_df = (
            pd.DataFrame(
                all_player_pts,
                columns=["name", "pid", "date", f"ff_pts_{scoring_source}"],
            )
            .set_index(["name", "pid"])
            .apply(pd.Series.explode)
            .reset_index()
            .clean_names()
        )
        self.ff_data = pd.merge(
            self.ff_data, all_pts_df, on=["name", "pid", "date"], how="inner"
        )
        logger.info(f"Fantasy points column '{self.ff_data.columns[-1]}' added")
        # return FantasyData

    def show_scoring_sources(self) -> List[str]:
        return list(self.scoring.keys())

    @property
    def data(self) -> pd.DataFrame:
        """Returns the dataframe of the historical NFL Fantasy data.

        Returns:
            pd.DataFrame: Historical NFL Fantasy data.
        """
        return self.ff_data

    def __str__(self) -> str:
        """Returns a string representation of the FantasyData object.

        Returns:
            str: Top 5 rows of the FantasyData object.
        """
        return str(self.ff_data.head())
