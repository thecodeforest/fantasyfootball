"""url: https://dailyroto.com/nfl-historical-production\
        -fantasy-points-draftkings-fanduel
"""

import pandas as pd
import pandas_flavor as pf
from janitor import clean_names, coalesce

from fantasyfootball.config import root_dir
from fantasyfootball.pipeline.utils import (
    get_module_purpose,
    map_abbr2_to_abbr3,
    map_player_names,
    read_args,
    read_ff_csv,
)


@pf.register_dataframe_method
def process_salary(df: pd.DataFrame, season_year: int) -> pd.DataFrame:
    column_mapping = {
        "player": "name",
        "p": "position",
        "player": "name",
        "opp_rank": "opp_position_rank",
        "opp_position_rank": "fanduel_salary",
        "fdsal": "draftkings_salary",
    }
    df = df.rename(columns=column_mapping)
    df = df[
        [
            "name",
            "position",
            "week",
            "team",
            "opp",
            "opp_position_rank",
            "fanduel_salary",
            "draftkings_salary",
        ]
    ]
    df["team"] = df["team"].apply(lambda x: map_abbr2_to_abbr3(x))
    df["opp"] = df["opp"].apply(lambda x: map_abbr2_to_abbr3(x))
    df["season_year"] = season_year
    return df


if __name__ == "__main__":
    args = read_args()
    dir_type, data_type = get_module_purpose(module_path=__file__)
    players_df = pd.read_csv(
        root_dir
        / "datasets"
        / "season"
        / str(args.season_year)
        / "processed"
        / "players"
        / "players.csv"
    )
    raw_data_dir = (
        root_dir
        / "staging_datasets"
        / "season"
        / str(args.season_year)
        / "raw"
        / data_type
    )
    clean_salary_df = read_ff_csv(raw_data_dir)
    clean_salary_df = (
        clean_salary_df.clean_names()
        .process_salary(season_year=args.season_year)
        .query("position != 'FB'")
    )
    player_name_map_df = map_player_names(
        players_df, clean_salary_df, "name", "position", "team"
    )
    clean_salary_df = (
        clean_salary_df.merge(
            player_name_map_df, on=["name", "position", "team"], how="left"
        )
        .coalesce("mapped_name", "name", target_column_name="final_name")
        .drop(columns=["name", "mapped_name"])
        .rename(columns={"final_name": "name"})
    )
    clean_salary_df = clean_salary_df[
        [clean_salary_df.columns.tolist()[-1]] + clean_salary_df.columns.tolist()[:-1]
    ]
    clean_salary_df.write_ff_csv(root_dir, args.season_year, dir_type, data_type)
