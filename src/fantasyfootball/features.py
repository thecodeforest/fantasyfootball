from __future__ import annotations  # noqa: F404

import logging
from itertools import product
from pathlib import PosixPath
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from fantasyfootball.config import data_sources, root_dir
from fantasyfootball.data import FantasyData

logger = logging.getLogger("fantasyfeatures")
logger.setLevel(logging.INFO)


class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    """Create lag features for each column in the dataframe by group.

    Args:
        n_week_lag (list): Number of weeks to lag the data
        lag_columns (list): Names of columns to lag
        player_group_columns (list): Names of columns to group by. For example,
            if you want to lag the data by player and season,
            you would pass in the list ["name", "season_year"]

    Returns:
        X (pd.DataFrame): Dataframe with lag features
    """

    def __init__(self, n_week_lag: list, lag_columns: list, player_group_columns: list):
        self.n_week_lag = n_week_lag
        self.lag_columns = lag_columns
        self.player_group_columns = player_group_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for col in self.lag_columns:
            for lag in self.n_week_lag:
                col_name = f"{col}_lag_{lag}"
                X = X.assign(
                    **{col_name: X.groupby(self.player_group_columns)[col].shift(lag)}
                )
        return X


class MAFeatureTransformer(BaseEstimator, TransformerMixin):
    """Create a moving average feature for each column in the dataframe by group

    Args:
        n_week_window (list): Number of weeks to average over
        window_columns (list): Names of columns to average over
        player_group_columns (list): Names of columns to group by. For example,
            if you want to lag the data by player and season,
            you would pass in the list ["name", "season_year"]

    Returns:
        X (pd.DataFrame): Dataframe with moving average features
    """

    def __init__(
        self, n_week_window: list, window_columns: list, player_group_columns: list
    ):
        self.n_week_window = n_week_window
        self.window_columns = window_columns
        self.player_group_columns = player_group_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for col in self.window_columns:
            for window in self.n_week_window:
                col_name = f"{col}_ma_{window}"
                X = X.assign(
                    **{
                        col_name: X.groupby(self.player_group_columns)[col].transform(
                            lambda x: x.rolling(
                                window, min_periods=1, center=False
                            ).mean()
                        )
                    }
                )
        return X


class CategoryConsolidatorFeatureTransformer(BaseEstimator, TransformerMixin):
    """Reduce the number of categories in a categorical column.

    Bins column values that fall below a threshold into a single 'other' category.

    Args:
        category_columns (list): Names of columns to consolidate
        threshold (float): Threshold for consolidating categories. For example,
            if you want to consolidate categories with less than 1% of the data,
            you would pass in the float 0.01.

    Returns:
        X (pd.DataFrame): Dataframe with consolidated categories
    """

    def __init__(self, category_columns: list, threshold: float):
        if isinstance(category_columns, str):
            category_columns = [category_columns]
        self.category_columns = category_columns
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        n = X.shape[0]
        for col in self.category_columns:
            category_count = (
                pd.DataFrame(X[col].value_counts())
                .reset_index()
                .rename(columns={"index": col, col: "count"})
                .sort_values("count", ascending=False)
            )
            category_count["pct_of_obs"] = category_count["count"] / n
            category_count[f"{col}_consolidated"] = category_count.apply(
                lambda row: row[col] if row["pct_of_obs"] > self.threshold else "other",
                axis=1,
            )
            category_count = category_count.drop(columns=["count", "pct_of_obs"])
            X = pd.merge(X, category_count, how="left", on=col)
            X = X.drop(columns=col)
            X = X.rename(columns={f"{col}_consolidated": col})
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)


class TargetEncoderFeatureTransformer(BaseEstimator, TransformerMixin):
    """Replace a categorical column with the average target value for each category.

    Args:
        category_columns (list): Names of columns to target encode.

    Returns:
        X (pd.DataFrame): Dataframe with target encoded columns

    """

    def __init__(self, category_columns: list):
        if isinstance(category_columns, str):
            category_columns = [category_columns]
        self.category_columns = category_columns

    # fit target encoder to x and y
    def fit(self, X, y):
        # Encode each element of each column
        self.category_mappings = dict()
        for column in self.category_columns:
            column_mappings = dict()
            unique_column_values = X[column].unique()
            for unique_value in unique_column_values:
                column_mappings[unique_value] = y[X[column] == unique_value].mean()
            self.category_mappings[column] = column_mappings
        return self

    def transform(self, X, y=None):
        for column, column_mappings in self.category_mappings.items():
            col_name = f"{column}_te"
            values = np.full(X.shape[0], np.nan)
            for value, mean_target in column_mappings.items():
                values[X[column] == value] = mean_target
            X = X.assign(**{col_name: values})
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)


class FantasyFeatures:
    """Create common fantasy football features for predictive modeling

    Args:
        df (pd.DataFrame): Dataframe containing player data by season
        position (str): Position of players to include in the dataframe
        player_group_columns (list, optional): Indicates which columns should
            be used to group players by. Defaults to
            ["pid", "name" ,"team" ,"season_year"].
        game_week_column (str, optional): Indicates week of season.
            Defaults to "week".

    Raises:
        ValueError: If position is not a valid position
        ValueError: If player_group_columns are not present in the dataframe
    """

    def __init__(
        self,
        df: pd.DataFrame,
        y: str,
        position: str,
        player_group_columns: list = ["pid", "name", "team", "season_year"],
        game_week_column: str = "week",
    ):
        if position not in ["QB", "RB", "WR", "TE"]:
            raise ValueError("Position must be one of QB, RB, WR, TE")
        logger.info(f"Filtering to {position}s.")
        # check if group columns are subset of df columns
        if not set(player_group_columns).issubset(set(df.columns)):
            raise ValueError("`player_group_columns` must be a subset of df columns")
        self.df = df[df["position"] == position].sort_values(
            player_group_columns + [game_week_column]
        )
        self.y = y
        self.position = position
        self.player_group_columns = player_group_columns
        self.game_week_column = game_week_column
        self.new_pipeline_features = list()
        self._pipeline_steps = ""

    @property
    def data(self) -> pd.DataFrame:
        """Returns the dataframe of the historical NFL Fantasy data.

        Returns:
            pd.DataFrame: Historical NFL Fantasy data.
        """
        return self.df

    def filter_inactive_games(
        self, status_column: str = "is_active"
    ) -> FantasyFeatures:
        """Filter out inactive games.

        Args:
            status_column (str, optional): Name of column indicating whether
            a player was active in a game. Defaults to "is_active".

        Returns:
            FantasyFeatures: FantasyFeatures object with inactive games removed.
        """
        if not all(x in [0, 1] for x in self.df[status_column]):
            raise ValueError(
                "status_column must be 0 or 1 indicating if player is active"
            )
        logger.info("Removing all rows where player was not active for game")
        self.df = self.df[self.df[status_column] == 1]
        logger.info(f"dropping {status_column} column")
        self.df = self.df.drop(columns=status_column)
        return FantasyFeatures

    @staticmethod
    def _calculate_n_games_played(
        df: pd.DataFrame, player_group_columns: list
    ) -> pd.DataFrame:
        """Calculate the number of games played by each player in each season.

        Helpful for filtering out players who have not played a certain number of
        games in a season, which can lead to issues when creating lag features.

        Args:
            df (pd.DataFrame): Dataframe to calculate the number of games
                played for each player.
            player_group_columns (list): Collection of columns that
                are unique to each player.

        Returns:
            pd.DataFrame: Dataframe with the number of games played for each
            player in each season.
        """
        # remove inactive games before calculating games played
        if "is_active" in df.columns:
            df = df[df["is_active"] == 1]

        games_played_this_season_df = (
            df.groupby(player_group_columns)
            .size()
            .to_frame("n_games_played")
            .reset_index()
        )
        # if future week has already been added
        # subtract 1 to ignore unplayed, future game
        if "is_future_week" in df.columns:
            games_played_this_season_df["n_games_played"] = (
                games_played_this_season_df["n_games_played"] - 1
            )

        return games_played_this_season_df

    def filter_n_games_played_by_season(self, min_games_played: int) -> FantasyFeatures:
        """Filter out players who have not played a certain number
        of games in a season.

        Args:
            min_games_played (int): Minimum number of games a player
            must have played in a season.

        Returns:
            FantasyFeatures: FantasyFeatures object with filtered dataframe.
        """
        games_played_this_season_df = self._calculate_n_games_played(
            self.df, self.player_group_columns
        )
        players_above_threshold_df = games_played_this_season_df.query(
            f"n_games_played >= {min_games_played}"
        ).drop(columns="n_games_played")
        self.df = pd.merge(
            self.df,
            players_above_threshold_df,
            on=self.player_group_columns,
            how="inner",
        )
        return FantasyFeatures

    @staticmethod
    def _save_pipeline_feature_names(
        columns: List[str], feature_type: str, *values: Union[int, str]
    ) -> List[str]:
        """Saves the names of the features created by the pipeline to a string.

        The feature names are saved in the format: <column_name>_<feature_type>_<value>
        For example:
            * is_active_lag_1
            * rush_yds_lag_4
            * passing_yds_ma_2

        Args:
            columns List[str]: List of column names to save.
            feature_type (str): Type of feature (e.g. lag, moving average, etc.).
            values (Union[int, str]): Additional parameters passed to the transformer.

        Returns:
            List[str]: List of feature names.
        """
        column_combo = list(
            product(
                columns, [feature_type], [str(x) for x in values] if values else [""]
            )
        )
        column_combo = [[y for y in x if y] for x in column_combo]
        column_names = ["_".join(x) for x in column_combo if x]
        return column_names

    @staticmethod
    def _validate_max_week(season_year: int, week_number: int) -> None:
        """Validates that the week number is not greater than the max
        week for the season.

        Args:
           season_year (int): Year of the season.
           week_number (int): Week number of the season.

        Raises:
            ValueError: If the week number is greater than the max week (18)
                for the season for games after 2020.
            ValueError: If the week number is greater than the max week (17)
                for the season for games prior to 2020.
        """
        if season_year > 2020 and week_number > 17:
            raise ValueError(
                "Cannot create future week when the max week number is greater than 17"
            )
        if season_year <= 2020 and week_number > 16:
            raise ValueError(
                "Cannot create future week when the max week number is greater than 16"
            )
        return None

    @staticmethod
    def _validate_future_data_is_present(
        ff_data_dir: PosixPath, max_week: int, data_sources: dict
    ) -> bool:
        """Validates that the future data is present for the upcoming week.
        For example, if it is week

        Args:
            ff_data_dir (PosixPath): Path to the directory containing the future data.
            max_week (int): Max week number for the season + 1. For example,
                if the max week is 8, then the future, yet to-be-played week is 9
            data_sources (dict): A dictionary indicating the names of
                the data sources used in the fantasyfootball package. Note that
                when validating the future data, only those data sources
                with 'is_forward_looking' set to True are checked.
                Example:
                data_sources = {
                            "calendar": {
                                "keys": ["team", "season_year"],
                                "cols": ["date", "week", "team", "opp",
                                         "is_away", "season_year"],
                                "is_required": True,
                                "is_forward_looking": False,
                            }}

        Raises:
            ValueError: If 'week' or 'date' is not present
            ValueError: If 'week' is not equal to max_week
        """
        future_week = max_week + 1
        calendar_df = pd.read_csv(ff_data_dir / "calendar.gz", compression="gzip")
        future_data_sources = [
            k for k in data_sources.keys() if data_sources[k]["is_forward_looking"]
        ]
        for data in future_data_sources:
            dataset_df = pd.read_csv(ff_data_dir / f"{data}.gz", compression="gzip")
            if "week" in dataset_df.columns:
                future_week_df = dataset_df.query(f"week == {future_week}")
            elif "date" in dataset_df.columns:
                future_week_df = pd.merge(
                    dataset_df,
                    calendar_df[calendar_df["week"] == future_week][["date"]],
                    on="date",
                    how="inner",
                )
            else:
                raise ValueError(f"{data} is missing a 'week' or 'date' column")
            if future_week_df.empty:
                raise ValueError(
                    f"No data for week {future_week} in {data}"
                    f"{data} is refreshed each week on Tuesday during season"
                )
        return True

    def log_transform_y(self) -> FantasyFeatures:
        """Log transform the y column.

        Args:
            None

        Returns:
            FantasyFeatures: FantasyFeatures object with log transformed y.

        """
        logger.info(f"Adding 1 and log transforming {self.y}")
        # convert any negative scores to 0
        self.df[self.y] = self.df[self.y].transform(lambda x: 0 if x < 0 else x)
        self.df[self.y] = self.df[self.y].transform(lambda x: np.log1p(x))

    def create_future_week(self) -> FantasyFeatures:
        """Creates a dataframe of future features for an upcoming NFL game week.
        For example, if 'Week 8' is the most recent completed set of games,
        a single row of features will be created for each player for 'Week 9'.

        returns:
            FantasyFeatures: Appends a dataframe of future features to the
            historical data.
        """
        current_season_year = max(self.df["season_year"])
        current_season_df = self.df[self.df["season_year"] == current_season_year]
        max_week = max(current_season_df[self.game_week_column])
        self._validate_max_week(season_year=current_season_year, week_number=max_week)
        ff_data_dir = root_dir / "datasets" / "season" / str(current_season_year)
        self._validate_future_data_is_present(ff_data_dir, max_week, data_sources)
        _load_data = FantasyData._load_data
        season_ff_data = _load_data(ff_data_dir, data_sources, "stats")
        future_week_df = season_ff_data[
            (season_ff_data[self.game_week_column] == max_week + 1)
            & (season_ff_data["season_year"] == current_season_year)
            & (season_ff_data["position"] == self.position)
        ]
        # load in historical stats data to add in player id
        stats_df = pd.read_csv(ff_data_dir / "stats.gz", compression="gzip")
        stats_df = stats_df[["name", "team", "pid"]].drop_duplicates()
        # assume player is active
        if "is_active" in self.df.columns:
            stats_df["is_active"] = 1
        # assume player is starting
        stats_df["is_start"] = 1
        future_week_df = pd.merge(
            future_week_df, stats_df, how="left", on=["name", "team"]
        )
        future_week_df["is_future_week"] = 1
        self.df = (
            pd.concat([self.df, future_week_df], axis=0)
            .sort_values(self.player_group_columns + [self.game_week_column])
            .reset_index(drop=True)
        )
        self.df["is_future_week"] = self.df["is_future_week"].fillna(0)
        return FantasyFeatures

    @staticmethod
    def _create_step_str(step: str, transformer_name: str, **params) -> str:
        """Creates a string representation of a pipeline step.

        Args:
            step (str): Description of what feature transformer is being used.
            transformer_name (str): Name of the feature transformer class.
            **params (dict): Parameters for the transformer.

        Returns:
            str: String representation of the pipeline step.
        """
        param_str = ", ".join(
            "{}={}".format(key, value) for key, value in params.items()
        )
        step_str = f"('{step}', {transformer_name}({param_str}))"
        return step_str

    def _validate_column_present(self, feature_columns: Union[str, list]) -> None:
        """Validates that a column is present in the dataframe prior to
        adding a new feature.

        Args:
            feature_columns (Union[str,list]): Columns to validate as present.

        Raises:
            ValueError: If any of the feature columns are not present.
        """
        if not isinstance(feature_columns, list):
            feature_columns = [feature_columns]
        for column in feature_columns:
            if column not in self.df.columns:
                raise ValueError(f"{column} not in dataframe")
        return True

    def add_lag_feature(
        self, n_week_lag: Union[int, List[int]], lag_columns: Union[str, List[str]]
    ) -> FantasyFeatures:
        """Adds string representation of a lag step to the pipeline.

        Args:
            n_week_lag (Union[int, List[int]]): Number of weeks to lag.
            lag_columns (Union[str, List[str]]): Columns to lag.

        Returns:
            FantasyFeatures: Updated string representation of the pipeline steps.
        """
        feature_type = "lag"
        if isinstance(n_week_lag, int):
            n_week_lag = [n_week_lag]
        if isinstance(lag_columns, str):
            lag_columns = [lag_columns]
        self._validate_column_present(feature_columns=lag_columns)
        new_lag_features = self._save_pipeline_feature_names(
            lag_columns, feature_type, *n_week_lag
        )
        self.new_pipeline_features = self.new_pipeline_features + new_lag_features
        lag_step_str = self._create_step_str(
            step="Create Lags of Features",
            transformer_name="LagFeatureTransformer",
            player_group_columns=self.player_group_columns,
            n_week_lag=n_week_lag,
            lag_columns=lag_columns,
        )

        logger.info("add lag step")
        self._pipeline_steps += lag_step_str + ","
        return FantasyFeatures

    def add_moving_avg_feature(
        self,
        n_week_window: Union[int, List[int]],
        window_columns: Union[str, List[str]],
    ) -> FantasyFeatures:
        """Adds string representation of a moving average step to the pipeline.

        Args:
            n_week_window (Union[int, List[int]]): Number of weeks to average across.
            window_columns (Union[str, List[str]]): Columns to average.

        Returns:
            FantasyFeatures: Updated string representation of the pipeline steps.
        """
        feature_type = "ma"
        if isinstance(n_week_window, int):
            n_week_window = [n_week_window]
        if isinstance(window_columns, str):
            window_columns = [window_columns]
        self._validate_column_present(feature_columns=window_columns)
        new_ma_features = self._save_pipeline_feature_names(
            window_columns, feature_type, *n_week_window
        )
        self.new_pipeline_features = self.new_pipeline_features + new_ma_features
        ma_step_str = self._create_step_str(
            step="Create Moving Average of Features",
            transformer_name="MAFeatureTransformer",
            player_group_columns=self.player_group_columns,
            n_week_window=n_week_window,
            window_columns=window_columns,
        )
        logger.info("add moving average")
        self._pipeline_steps += ma_step_str + ","
        return FantasyFeatures

    def add_target_encoded_feature(
        self, category_columns: Union[str, list]
    ) -> FantasyFeatures:
        """Adds string representation of a target encoded step to the pipeline.

        Args:
            category_columns (Union[str, list]): Columns to target encode.

        Returns:
            FantasyFeatures: Updated string representation of the pipeline steps.
        """
        feature_type = "te"
        if isinstance(category_columns, str):
            category_columns = [category_columns]
        self._validate_column_present(feature_columns=category_columns)
        new_te_feature = self._save_pipeline_feature_names(
            category_columns, feature_type
        )
        self.new_pipeline_features = self.new_pipeline_features + new_te_feature
        te_step_str = self._create_step_str(
            step="Target Encode Categorical Feature",
            transformer_name="TargetEncoderFeatureTransformer",
            category_columns=category_columns,
        )
        logger.info("add target encoding for categorical variables")
        self._pipeline_steps += te_step_str + ","
        return FantasyFeatures

    def consolidate_category_feature(
        self, category_columns: Union[str, list], threshold: float
    ) -> FantasyFeatures:
        """Adds string representation of a category consolidator step to the pipeline.

        Args:
            category_columns (Union[str, list]): Columns to consolidate.
            threshold (float): Threshold for consolidating categories.

        Returns:
            FantasyFeatures: Updated string representation of the pipeline steps.
        """
        if isinstance(category_columns, str):
            category_columns = [category_columns]
        self._validate_column_present(feature_columns=category_columns)
        cc_step_str = self._create_step_str(
            step="Consolidate Categorical Feature",
            transformer_name="CategoryConsolidatorFeatureTransformer",
            category_columns=category_columns,
            threshold=threshold,
        )
        logger.info("Consolidating levels for categorical variables")
        self._pipeline_steps += cc_step_str + ","
        return FantasyFeatures

    def _remove_missing_feature_values(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Removes rows that have missing values related to lag or salary columns.

        When creating a lag, the first N weeks of data will be NA. This function
        removes those rows.
        Likewise, Draftkings and Fanduel do not publish salary data for the
        first week of each season. This function removes those rows if
        any fields from the salary data are included.
        If both salary and lag data are included, the maximum of the two
        will be used when removing rows.

        Args:
            feature_df (pd.DataFrame): Dataframe to remove rows from.

        Returns:
            pd.DataFrame: Dataframe with missing lag values or salary data removed.
        """
        # max weeks to drop conditions.
        weeks_to_drop_lag = 0
        weeks_to_drop_salary = 0
        lag_fields = [x for x in feature_df.columns if "lag" in x]
        if lag_fields:
            weeks_to_drop_lag = max([int(x.split("_")[-1]) for x in lag_fields])
        salary_fields = [x for x in feature_df.columns if "salary" in x]
        # always drop week 1 if salary data is included, since it is not published
        if salary_fields:
            weeks_to_drop_salary = 1
        weeks_to_drop = max(weeks_to_drop_lag, weeks_to_drop_salary)
        feature_df["player_game_index"] = (
            feature_df.sort_values(self.player_group_columns + [self.game_week_column])
            .groupby(self.player_group_columns)
            .cumcount()
            .reset_index(drop=True)
        )
        feature_df = feature_df.query(f"player_game_index >= {weeks_to_drop}").drop(
            columns="player_game_index"
        )
        return feature_df

    def _replace_missing_salary_values_with_zero(
        self, feature_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Replaces missing salary values with zero.

        Salary data is missing for the first week of each season. Also,
        when players are injured and questionable to play, a salary
        value may not be published.

        Args:
            feature_df (pd.DataFrame): Dataframe to replace missing
            salary values with zero.

        Returns:
            pd.DataFrame: Dataframe with missing salary values replaced with zero.
        """
        salary_columns = [x for x in feature_df.columns if "salary" in x]
        if salary_columns:
            for column in salary_columns:
                feature_df[column] = feature_df[column].fillna(0)
            return feature_df
        else:
            return feature_df

    def _forward_fill_future_week_cv(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Forward fills CV values for future weeks."""
        if "cv" in feature_df.columns and "is_future_week" in feature_df.columns:
            feature_df["cv_ff"] = feature_df.groupby("pid")["cv"].transform(
                lambda x: x.ffill()
            )
            feature_df["cv"] = feature_df.apply(
                lambda row: row["cv"] if row["is_future_week"] == 0 else row["cv_ff"],
                axis=1,
            )
            feature_df = feature_df.drop(columns="cv_ff")
            return feature_df
        else:
            return feature_df

    def add_coefficient_of_variation(self, n_week_window: int) -> FantasyFeatures:
        """Add coefficient of variation (cv) for each player based
        the trailing standard deviation and average of weekly
        fantasy points scored.

        Args:
            n_week_window (int): Number of trailing weeks to
            use for calculating the cv. Note that calculation
            occurs across seasons.

        Returns:
            FantasyFeatures: Dataframe with cv added as a column.

        """
        keys = ["pid", "date"]
        cv_df = self.df[keys + [self.y]]
        cv_df = cv_df.sort_values(by=keys).reset_index(drop=True)
        # replace any negative point values with zero when calculating cv
        cv_df[self.y] = cv_df[self.y].apply(lambda x: 0 if x < 0 else x)
        sd = cv_df.groupby("pid")[self.y].apply(
            lambda x: x.rolling(n_week_window).std()
        )
        mu = cv_df.groupby("pid")[self.y].apply(
            lambda x: x.rolling(n_week_window).mean()
        )
        cv_df["cv"] = (sd / mu) * 100
        # replace any inf values with nan
        cv_df["cv"] = cv_df["cv"].apply(lambda x: np.nan if np.isinf(x) else x)
        cv_df["cv"] = cv_df["cv"].apply(
            lambda x: round(x) if not np.isnan(x) else np.nan
        )
        cv_df = cv_df.drop(columns=self.y)
        self.df = pd.merge(self.df, cv_df, on=keys, how="inner")
        return FantasyFeatures

    def create_ff_signature(self) -> dict:
        """Creates a fantasy football 'signature', which includes the following steps:

            * Executes the previously created pipeline data transformations
            * Removes missing values stemming from lagged features or salary features
            * Replaces missing salary values with zero

        Returns:
            dict: The names of the new features created by the pipeline and the
            transformed dataframe.
        """
        all_feature_steps = "[" + self._pipeline_steps + "]"
        pipeline = Pipeline(steps=eval(all_feature_steps))
        feature_df = pipeline.fit_transform(self.df, y=self.df[self.y])
        feature_df = self._remove_missing_feature_values(feature_df)
        if "salary" in feature_df.columns:
            feature_df = self._replace_missing_salary_values_with_zero(feature_df)
        # carry forward cv for each player to future week if cv in columns
        if "cv" in feature_df.columns:
            feature_df = self._forward_fill_future_week_cv(feature_df)
        return {
            "pipeline_feature_names": self.new_pipeline_features,
            "feature_df": feature_df,
        }
