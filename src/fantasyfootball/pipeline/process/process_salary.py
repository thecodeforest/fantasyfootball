import pandas as pd
import pandas_flavor as pf
from janitor import clean_names

from fantasyfootball.config import root_dir
from fantasyfootball.pipeline.utils import get_module_purpose, read_args, read_ff_csv


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
    return df


if __name__ == "__main__":
    args = read_args()
    dir_type, data_type = get_module_purpose(module_path=__file__)
    raw_data_dir = (
        root_dir / "datasets" / "season" / str(args.season_year) / "raw" / data_type
    )
    clean_salary_df = read_ff_csv(raw_data_dir)
    clean_salary_df = clean_salary_df.clean_names().process_salary(
        season_year=args.season_year
    )
    clean_salary_df.write_ff_csv(root_dir, args.season_year, dir_type, data_type)
