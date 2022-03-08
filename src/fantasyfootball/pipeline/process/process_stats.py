from fantasyfootball.config import root_dir
import pandas as pd
from fantasyfootball.pipeline.pipeline_logger import logger
from fantasyfootball.pipeline.utils import read_args, get_module_function, write_ff_csv
import pandas_flavor as pf
from janitor import clean_names
import re
from itertools import chain


REQUIRED_COLUMNS = {"player_columns": ["pid", "name"], 
                   "game_columns": ["tm", "opp", "status", "date", "result", "away", "gs"], 
                   "stats_columns": ["g_nbr", "receiving_rec", "receiving_yds", "receiving_td", 
                                    "rushing_yds", "rushing_td", 
                                    "passing_cmp", "passing_yds", "passing_td", 
                                    "fumbles_fmb", "passing_int"]
                                    }
@pf.register_dataframe_method
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the column names of the given dataframe.

    Args:
        df (pd.DataFrame): [description]

    Returns:
        pd.DataFrame: [description]
    """        
    df = clean_names(df)
    cols = df.columns
    cols = [re.sub('unnamed_[0-9]+_level_\d','',x).strip("_") for x in cols]
    # away will always be the first empty string following cleaning step above
    cols[cols.index("")] = "away"
    cols = [x.replace("#", "_nbr") for x in cols]
    cols = [x.replace("%", "_pct") for x in cols]
    df.columns = cols
    return df


@pf.register_dataframe_method
def clean_status_column(df: pd.DataFrame) -> pd.DataFrame:
    # empty string indicates player played, otherwise player was injured, sick, etc.
    if 'status' not in df.columns:
        df['status'] =  [1] * df.shape[0]
    else:
        df['status'] = [1 if not x else 0 for x in df['status']]
    return df

# TO DO: TEST THIS TO SEE HOW IT WORKS
@pf.register_dataframe_method
def add_missing_stats_columns(df: pd.DataFrame, required_columns: dict = REQUIRED_COLUMNS) -> pd.DataFrame:
    df = df.copy()
    current_columns = df.columns.tolist()
    missing_columns = set(chain(*required_columns.values())) - set(current_columns)
    if missing_columns:
        for column in missing_columns:
            df[column] = 0
    return df


@pf.register_dataframe_method
def select_ff_columns(df: pd.DataFrame, required_columns: dict = REQUIRED_COLUMNS) -> pd.DataFrame:
    return df[required_columns["player_columns"] + 
              required_columns["game_columns"] + 
              required_columns["stats_columns"]
              ]


def clean_stats_column(col: pd.Series) -> list:
    col = col.fillna("0")
    col = col.astype(str).tolist()
    col = [x.strip('%') for x in col]
    numeric_values = [x.replace('.', '').isdigit() for x in col]
    if not all(numeric_values):
        non_numeric_value_indexes = [index for (index, value) in enumerate(numeric_values) if not value]
        for i in non_numeric_value_indexes: col[i] = "0"
    col = [float(x) for x in col]
    return col


@pf.register_dataframe_method
def clean_stats_column_values(df: pd.DataFrame, stats_columns: list = REQUIRED_COLUMNS["stats_columns"]) -> pd.DataFrame:
    df = df.copy()
    for col in stats_columns:
        df[col] = clean_stats_column(col=df[col])
    return df


@pf.register_dataframe_method
def recode_str_to_numeric(df: pd.DataFrame, column: str, target_value: str, replacement_value: int) -> pd.DataFrame:
    if replacement_value not in [0, 1]:
        raise ValueError("replacement_value must be 0 or 1")
    other_replacement_value = int(not replacement_value)
    df = df.assign(**{column: df[column].transform(lambda x: replacement_value if x == target_value else other_replacement_value)})
    return df
    

if __name__ == "__main__":
    args = read_args()
    dir_type, data_type = get_module_function(module_path=__file__)
    raw_data_dir = root_dir / "data" / "season" / str(args.season_year) / "raw"  / data_type
    raw_stats_files = raw_data_dir.glob("*.csv")
    for player_stats_path in raw_stats_files:
        clean_stats_df = pd.read_csv(player_stats_path, keep_default_na=False)   
        pid = clean_stats_df["pid"].iloc[0] 
        logger.info(f"Processing player {pid}")
        clean_stats_df = (clean_stats_df.clean_column_names()
                                        .clean_status_column()
                                        .add_missing_stats_columns()
                                        .select_ff_columns()
                                        .clean_stats_column_values()
                                        .query("~date.str.contains('Games')")
                                        .recode_str_to_numeric(column="away", target_value="@", replacement_value=1)
                                        .recode_str_to_numeric(column="gs", target_value="*", replacement_value=1)
                                        
                          )                                       
        clean_stats_df.write_ff_csv(root_dir, 
                                args.season_year,
                                dir_type,
                                data_type,
                                pid
                                )     



