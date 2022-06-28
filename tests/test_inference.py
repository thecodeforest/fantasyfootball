import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from fantasyfootball.config import root_dir, data_sources, scoring
from fantasyfootball.data import FantasyData
from fantasyfootball.features import FantasyFeatures

INPUT_FEATURES = {
    "QB": {
        "raw_features": ["draftkings_salary", "avg_windspeed", "avg_temp"],
        "position_features": ["passing_cmp"],
    },
    "RB": {
        "raw_features": ["draftkings_salary", "fanduel_salary"],
        "position_features": ["rushing_yds"],
    },
    "WR": {
        "raw_features": ["draftkings_salary", "has_dnp_tag"],
        "position_features": ["receiving_rec"],
    },
    "TE": {
        "raw_features": ["draftkings_salary", "fanduel_salary"],
        "position_features": ["receiving_rec"],
    },
}


def find_most_recent_season() -> int:
    datasets_dir = root_dir / "datasets" / "season"
    all_seasons = sorted(
        [x.stem for x in list(datasets_dir.iterdir()) if x.stem.isalnum()], reverse=True
    )
    for season_year in all_seasons:
        all_datasets = list(Path(datasets_dir / season_year).glob("*.gz"))
        if "stats" in [x.stem for x in all_datasets]:
            return int(season_year)


def find_most_recent_complete_week(df: pd.DataFrame, most_recent_season: int) -> int:
    game_count_by_week = (
        df.query(f"season_year=={most_recent_season}")[["week", "team"]]
        .drop_duplicates()
        .groupby("week")["team"]
        .count()
        .reset_index()
        .rename(columns={"team": "count"})
        .query("count > 26")
    )
    most_recent_complete_week = max(game_count_by_week["week"])
    return most_recent_complete_week


def test_inference():
    exp_mae_lb = 2
    exp_mae_ub = 10
    exp_r2_lb = 0.1
    exp_r2_ub = 0.6

    scoring_source = "yahoo"
    most_recent_season = find_most_recent_season()
    fantasy_data = FantasyData(2016, most_recent_season)
    fantasy_data.create_fantasy_points_column(scoring_source=scoring_source)
    fantasy_df = fantasy_data.data
    yvar = fantasy_df.columns[-1]
    yvar_pred = f"{yvar}_pred"
    most_recent_complete_week = find_most_recent_complete_week(
        fantasy_df, most_recent_season
    )
    test_df = fantasy_df.query(
        f"season_year=={most_recent_season} & week == {most_recent_complete_week}"
    )
    train_df = fantasy_df[fantasy_df["date"] < min(test_df["date"])]
    for position in INPUT_FEATURES.keys():
        ff = FantasyFeatures(train_df, yvar, position)
        ff.log_transform_y()
        ff.filter_inactive_games(status_column="is_active")
        ff.create_future_week()
        ff.filter_n_games_played_by_season(min_games_played=2)
        ff.add_lag_feature(
            n_week_lag=1,
            lag_columns=[yvar] + INPUT_FEATURES.get(position).get("position_features"),
        )
        ff.add_moving_avg_feature(
            n_week_window=4,
            window_columns=[yvar]
            + INPUT_FEATURES.get(position).get("position_features"),
        )
        ff_sig_dict = ff.create_ff_signature()
        derived_features = ff_sig_dict.get("pipeline_feature_names")
        feature_df = ff_sig_dict.get("feature_df")
        # filter to "train" (past week) from "test" (future week)
        hist_df = feature_df[feature_df["is_future_week"] == 0]
        future_df = feature_df[feature_df["is_future_week"] == 1]
        #
        all_features = (
            INPUT_FEATURES.get(position).get("raw_features") + derived_features
        )
        # split historical X and y
        X_hist, y_hist = hist_df[all_features + [yvar]].get_features_targets(
            yvar, all_features
        )
        # get future X
        X_future = future_df[["pid"] + all_features]
        reg = LinearRegression().fit(X_hist.values, y_hist.values)
        preds = [np.expm1(x) for x in reg.predict(X_future.drop(columns="pid").values)]
        X_future = X_future.assign(**{yvar_pred: preds})
        # compare predictions to actuals
        results = pd.merge(test_df[["pid"] + [yvar]], X_future, on="pid", how="inner")
        results = pd.merge(results, test_df.query("is_active == 1")[["pid"]])
        results = results[["pid", yvar, yvar_pred]]

        result_mae = mean_absolute_error(results[yvar], results[yvar_pred])
        result_r2 = round(
            np.corrcoef(results[yvar].tolist(), results[yvar_pred].tolist())[0][1] ** 2,
            3,
        )

        assert (
            exp_mae_lb <= result_mae <= exp_mae_ub
        ), f"MAE {result_mae} not in range {exp_mae_lb}-{exp_mae_ub} for {position}"
        assert (
            exp_r2_lb <= result_r2 <= exp_r2_ub
        ), f"R-Squared {result_r2} not in range {exp_r2_lb}-{exp_r2_ub} for {position}"
