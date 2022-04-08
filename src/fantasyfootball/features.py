from itertools import product
from typing import List, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from fantasyfootball.config import root_dir


class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    """Create a lag feature for each column in the dataframe by group.

    Args:
        n_week_lag (list): list of integers representing the number of weeks to lag
        lag_columns (list): list of column names to lag
        player_group_columns (list): list of column names to group by
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
        n_week_window (list): list of integers representing
            the number of weeks to window.
        window_columns (list): list of column names to window
        player_group_columns (list): list of column names to group by

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


class FantasyFeatures:
    """Create common fantasy football features for predictive modeling"""

    def __init__(
        self,
        df: pd.DataFrame,
        position: str,
        player_group_columns: list = ["pid", "name", "team", "season_year"],
        game_week_column: str = "week",
    ):
        """

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
        if position not in ["QB", "RB", "WR", "TE"]:
            raise ValueError("Position must be one of QB, RB, WR, TE")
        print(f"Filtering to {position}s.")
        # check if group columns are subset of df columns
        if not set(player_group_columns).issubset(set(df.columns)):
            raise ValueError("`player_group_columns` must be a subset of df columns")
        self.df = df[df["position"] == position].sort_values(
            player_group_columns + [game_week_column]
        )
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
        games_played_this_season_df = (
            df.groupby(player_group_columns)
            .size()
            .to_frame("n_games_played")
            .reset_index()
        )
        return games_played_this_season_df

    def filter_n_games_played_by_season(self, min_games_played: int) -> None:
        """Filter out players who have not played a certain number
           of games in a season.

        Args:
            min_games_played (int): Minimum number of games a player
            must have played in a season.

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
        n_players_removed = (
            games_played_this_season_df.shape[0] - players_above_threshold_df.shape[0]
        )
        print(
            f"{n_players_removed} player(s) removed who"
            f"have not played {min_games_played} games in a season"
        )

    @staticmethod
    def _format_pd_filter_query(columns: List[str], row_values: List[str]) -> List[str]:
        """Helper function to format a query to filter out rows
        Args:
            columns List[str] : Column names to filter by
            row_values List[str]: Pandas row as a list
        Returns:
            List[str]: Query to filter out rows
        """
        query = list(zip(columns, row_values))
        value_types = [type(x[1]) for x in query]
        query = [[x[0], str(x[1])] for x in query]
        # add "" if type of value is string
        for index, value in enumerate(value_types):
            if value is str:
                query[index][1] = '"' + query[index][1] + '"'
        # add equality sign to query
        query = ["==".join(x) for x in query]
        if len(query) > 1:
            query = " and ".join(query)
        else:
            query = query[0]
        return query

    @staticmethod
    def _save_pipeline_feature_names(
        columns: List[str], feature_type: str, *values: Union[int, str]
    ) -> List[str]:
        """Saves the names of the features created by the pipeline to a string.
           The feature names are saved in the format:
              <column_name>_<feature_type>_<value>
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

    # TO DO: Refactor this method into smaller methods
    def create_future_week(self):
        """Creates a dataframe of future features for an upcoming NFL game week.
        For example, if 'Week 8' is the most recent completed set of games,
        a single row of features will be created for each player for 'Week 9'.

        The future features include:
           * The defensive strength of the upcoming opponent against
             the passing, rushing, and receiving.
           * The projected number of offensive points for each team.
             Point projections are calculated based on the opening spread
             and the over/under.
           * The forecasted daily weather for the game location.
        """
        current_season_year = max(self.df["season_year"])
        current_season_df = self.df[self.df["season_year"] == current_season_year]
        max_week = max(current_season_df[self.game_week_column])
        self._validate_max_week(season_year=current_season_year, week_number=max_week)
        ff_data_dir = root_dir.parent.parent / "datasets" / "season"
        calendar_df = pd.read_csv(
            ff_data_dir / str(current_season_year) / "calendar.csv"
        ).drop(columns="season_year")
        betting_df = pd.read_csv(ff_data_dir / str(current_season_year) / "betting.csv")
        defense_df = pd.read_csv(ff_data_dir / str(current_season_year) / "defense.csv")
        weather_df = pd.read_csv(ff_data_dir / str(current_season_year) / "weather.csv")
        future_week_df = pd.DataFrame()
        for row in (
            current_season_df[self.player_group_columns]
            .drop_duplicates()
            .itertuples(index=False)
        ):
            player_filter_query = self._format_pd_filter_query(
                columns=self.player_group_columns, row_values=list(row)
            )
            player_df = current_season_df.query(player_filter_query)
            upcoming_game_df = calendar_df[
                (calendar_df[self.game_week_column] == max_week + 1)
                & (calendar_df["team"] == player_df["team"].iloc[0])
            ]
            if upcoming_game_df.empty:  # indicates buy week if empty
                continue
            upcoming_game_defense = defense_df[
                (defense_df["week"] == max_week)
                & (defense_df["opp"] == upcoming_game_df["opp"].iloc[0])
            ].drop(columns="week")
            upcoming_game_df = pd.merge(
                upcoming_game_df, upcoming_game_defense, on="opp", how="inner"
            )
            # add in point projections based on over/under and point spread
            upcoming_game_df = pd.merge(
                upcoming_game_df,
                betting_df.drop(columns="season_year"),
                on=["team", "opp", "date"],
                how="inner",
            )
            # add in weather data (will use forecast for future, unplayed games;
            # otherwise uses actual historical weather)
            upcoming_game_df = pd.merge(
                upcoming_game_df, weather_df, on=["date", "team", "opp"], how="inner"
            )
            player_attributes = pd.DataFrame(
                {
                    **dict(zip(self.player_group_columns, list(row))),
                    **{"status": 1, "position": self.position},
                },
                index=[0],
            )
            upcoming_game_df = pd.concat(
                [
                    player_attributes.drop(columns=["team", "season_year"]),
                    upcoming_game_df,
                ],
                axis=1,
            )
            future_week_df = future_week_df.append(upcoming_game_df)
        # ensure no features exist in future week that are not present in df
        future_week_df = future_week_df[
            list(set(self.df.columns).intersection(future_week_df.columns))
        ]
        future_week_df["is_future_week"] = 1
        self.df = (
            pd.concat([self.df, future_week_df], axis=0)
            .sort_values(self.player_group_columns + [self.game_week_column])
            .reset_index(drop=True)
        )
        self.df["is_future_week"] = self.df["is_future_week"].fillna(0)

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

    def _validate_n_week_lag_length(self, n_week_lag: list):
        max_lag_length = max(n_week_lag)
        games_played_this_season_df = self._calculate_n_games_played(
            self.df, self.player_group_columns
        )
        min_games_played = min(games_played_this_season_df["n_games_played"])
        # exclude future week from count if present
        if "is_future_week" in self.df.columns:
            min_games_played -= 1
        if max_lag_length > min_games_played:
            raise ValueError(
                f"The lag length of {max_lag_length}\n"
                "is greater than the minimum number of games\n"
                f"played in the season ({min_games_played}) for some players.\n"
                "Either reduce lag length or ensure players have played\n"
                f"at least {max_lag_length} games"
            )

    def _validate_column_present(self, feature_columns: list) -> None:
        for column in feature_columns:
            if column not in self.df.columns:
                raise ValueError(f"{column} not in dataframe")

    def add_lag_feature(self, n_week_lag: list, lag_columns: list):
        feature_type = "lag"
        self._validate_column_present(feature_columns=lag_columns)
        self._validate_n_week_lag_length(self.df, self.player_group_columns, n_week_lag)
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

        print("add lag step")
        self._pipeline_steps += lag_step_str + ","

    def add_moving_avg_feature(self, n_week_window: list, window_columns: list):
        feature_type = "ma"
        self._validate_column_present(feature_columns=window_columns)
        new_ma_features = self._save_pipeline_feature_names(
            window_columns, feature_type, *n_week_window
        )
        self.new_pipeline_features = self.new_pipeline_features + new_ma_features
        ma_step_str = self._create_step_str(
            step="Create Moving Aveage of Features",
            transformer_name="MAFeatureTransformer",
            player_group_columns=self.player_group_columns,
            n_week_window=n_week_window,
            window_columns=window_columns,
        )
        print("add moving average")
        self._pipeline_steps += ma_step_str + ","

    def create_ff_signature(self):
        all_feature_steps = "[" + self._pipeline_steps + "]"
        pipeline = Pipeline(steps=eval(all_feature_steps))
        ff_df_trans = pipeline.fit_transform(self.df)
        return self.new_pipeline_features, ff_df_trans
