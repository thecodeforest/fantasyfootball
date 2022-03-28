import numpy as np
import pandas as pd
from janitor import clean_names

from fantasyfootball.config import root_dir


class FantasyData:
    """Collects and loads all historical fantasy football datasets
       based on the range years provided. Currently, the following
       weekly data sources are included:
         * Player stats
         * Betting data, which includes opening point spreads and over-under lines
         * Defense data, which ranks the strength of the opposing team's defense
       See the README for more details on how to interpret each field.

    Args:
        season_year_start (int): The first year of the season.
        season_year_end (int): The last year of the season.
        ff_data (pd.DataFrame): The dataframe containing all historical
        fantasy football data.
        scoring (dict): The scoring rules for calculating weekly fantasy points.

    """

    def __init__(self, season_year_start: int, season_year_end: int) -> None:
        self.season_year_start = season_year_start
        self.season_year_end = season_year_end
        self.ff_data = None
        self.validate_season_year_range()
        self.load_data()
        self.scoring = {
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

    def validate_season_year_range(self) -> None:
        """Ensures that the season year range is valid.

        Raises:
            ValueError: If the season year is less than the minimum year
            ValueError: If the season year is greater than the maximum year
        """
        season_years = [
            int(str(x).split("/")[-1])
            for x in (root_dir.parent.parent / "datasets" / "season").glob("*")
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

    def load_data(self) -> None:
        """Loads all historical fantasy football data from the season year
        range provided. Each season year is loaded separately and
        then concatenated together.
        """
        print(f"Loading data from {self.season_year_start} to {self.season_year_end}")
        ff_data_dir = root_dir.parent.parent / "datasets" / "season"
        ff_df = pd.DataFrame()
        for season_year in range(self.season_year_start, self.season_year_end + 1):
            ff_season_data_dir = ff_data_dir / str(season_year)
            calendar_df = pd.read_csv(ff_season_data_dir / "calendar.csv")
            players_df = pd.read_csv(ff_season_data_dir / "players.csv")
            season_ff_df = pd.merge(
                calendar_df, players_df, how="inner", on=["team", "season_year"]
            )
            stats_df = pd.read_csv(ff_season_data_dir / "stats.csv").drop(
                columns="away"
            )
            season_ff_df = pd.merge(
                season_ff_df, stats_df, how="inner", on=["date", "name", "team", "opp"]
            )
            betting_df = pd.read_csv(ff_season_data_dir / "betting.csv")
            season_ff_df = pd.merge(
                season_ff_df,
                betting_df,
                how="left",
                on=["date", "season_year", "team", "opp"],
            )
            defense_df = pd.read_csv(ff_season_data_dir / "defense.csv")
            season_ff_df = pd.merge(
                season_ff_df, defense_df, how="inner", on=["week", "opp"]
            )
            ff_df = ff_df.append(season_ff_df)
        self.ff_data = ff_df

    @staticmethod
    def validate_scoring_source_rules(source_rules: dict, ff_df_columns: list) -> None:
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

    def add_scoring_source(self, source_rules: dict) -> None:
        """Updates the scoring source rules.

        Args:
            source_rules (dict): Scoring source rules. Required keys
            are the source name (e.g., 'custom'), 'scoring_columns', and 'multiplier'.

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
        self.validate_scoring_source_rules(source_rules, self.ff_data.columns.tolist())
        source_name = list(source_rules.keys())[0]
        self.scoring[source_name] = source_rules
        print(f"Scoring source '{source_name}' Added")

    def create_fantasy_points_column(self, scoring_source: str) -> None:
        """Creates a fantasy points column for the scoring source provided.

        Args:
            scoring_source (str): Name of the scoring source to use
            (e.g., 'draft kings', 'yahoo', 'custom').

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
            player_weekly_points = [0] * player_df.shape[0]
            for column in scoring_columns:
                point_amount = scoring_source_rules["scoring_columns"][column]
                scoring_amount = player_df[column]
                weekly_points_scored = [x * point_amount for x in scoring_amount]
                player_weekly_points = np.add(
                    player_weekly_points, weekly_points_scored
                )
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
        print(f"Fantasy points column '{self.ff_data.columns[-1]}' added")

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
