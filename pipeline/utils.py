import argparse
from pathlib import Path, PosixPath
from datetime import datetime, timedelta
from typing import List, Tuple, Union

import pandas as pd
import pandas_flavor as pf
from fuzzywuzzy import fuzz

TEAM_ABBREVIATION_MAPPING = {
    ("Arizona", "Arizona Cardinals", "Cardinals"): "ARI",
    ("Atlanta", "Atlanta Falcons", "Falcons"): "ATL",
    ("Buffalo", "Buffalo Bills", "Bills"): "BUF",
    ("Baltimore", "Baltimore Ravens", "Ravens"): "BAL",
    ("Carolina", "Carolina Panthers", "Panthers"): "CAR",
    ("Chicago", "Chicago Bears", "Bears"): "CHI",
    ("Cincinnati", "Cincinnati Bengals", "Bengals"): "CIN",
    ("Cleveland", "Cleveland Browns", "Browns"): "CLE",
    ("Dallas", "Dallas Cowboys", "Cowboys"): "DAL",
    ("Denver", "Denver Broncos", "Broncos"): "DEN",
    ("Detroit", "Detroit Lions", "Lions"): "DET",
    ("GreenBay", "Green Bay Packers", "Packers"): "GNB",
    ("Houston", "Houston Texans", "Texans"): "HOU",
    ("Indianapolis", "Indianapolis Colts", "Colts"): "IND",
    ("Jacksonville", "Jacksonville Jaguars", "Jaguars"): "JAX",
    ("KansasCity", "Kansas City Chiefs", "KCChiefs", "Kansas", "Chiefs"): "KAN",
    ("LAChargers", "Los Angeles Chargers", "LosAngeles", "Chargers"): "LAC",
    ("Oakland", "Oakland Raiders"): "OAK",
    ("LARams", "Los Angeles Rams", "Rams"): "LAR",
    ("LasVegas", "Las Vegas Raiders", "LVRaiders", "Raiders"): "LVR",
    ("Miami", "Miami Dolphins", "Dolphins"): "MIA",
    ("Minnesota", "Minnesota Vikings", "Vikings"): "MIN",
    ("NewEngland", "New England Patriots", "Patriots"): "NWE",
    ("NewOrleans", "New Orleans Saints", "Saints"): "NOR",
    ("NYGiants", "New York Giants", "Giants"): "NYG",
    ("NYJets", "New York Jets", "Jets"): "NYJ",
    ("Philadelphia", "Philadelphia Eagles", "Eagles"): "PHI",
    ("Pittsburgh", "Pittsburgh Steelers", "Steelers"): "PIT",
    ("San Diego", "San Diego Chargers", "SanDiego", "Chargers"): "SDG",
    ("SanFrancisco", "San Francisco 49ers", "49ers"): "SFO",
    ("St Louis", "St. Louis Rams", "St.Louis", "Rams"): "STL",
    ("Seattle", "Seattle Seahawks", "Seahawks"): "SEA",
    ("TampaBay", "Tampa Bay Buccaneers", "Tampa", "Buccaneers"): "TAM",
    ("Tennessee", "Tennessee Titans", "Titans"): "TEN",
    (
        "Washington",
        "Washingtom",
        "Washington Football Team",
        "Washington Redskins",
        "Washington Commanders",
        "Commanders",
    ): "WAS",
}


def read_args() -> argparse.Namespace:
    """Helper function to read command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--season_year", type=int, help="The season year")
    parser.add_argument("--s3_bucket", type=str, help="The S3 bucket to write to")
    parser.add_argument(
        "--open_weather_api_key", type=str, help="The S3 bucket to write to"
    )
    parser.add_argument(
        "--is_historical", type=bool, help="If the data is historical or future looking"
    )
    args = parser.parse_args()
    return args


def retrieve_team_abbreviation(team_name: str) -> str:
    """Retrieves the team abbreviation from the team name.

    Args:
        team_name (str): The full team name or team's home city.

    Raises:
        ValueError: If team name is not found in the team_abbreviation_mapping dict.

    Returns:
        str: The team abbreviation.
    """
    team_abbreviation_mapping = TEAM_ABBREVIATION_MAPPING
    for k, v in team_abbreviation_mapping.items():
        if team_name in k:
            return v
    raise ValueError(f"Team name {team_name} not found in team_abbreviation_mapping")


def map_abbr2_to_abbr3(team_name: str) -> str:
    "Maps the 2-letter abbreviation to the 3-letter abbreviation."
    abbr2_mapping = {
        "KC": "KAN",
        "LA": "LAC",
        "LV": "LVR",
        "SF": "SFO",
        "TB": "TAM",
        "GB": "GNB",
        "NE": "NWE",
        "NO": "NOR",
        "PH": "PHI",
        "SD": "SDG",
        "NO": "NOR",
    }
    if len(team_name) == 2:
        if team_name not in abbr2_mapping.keys():
            raise ValueError(f"{team_name} not found in abbr2_mapping")
        return abbr2_mapping[team_name]
    else:
        return team_name


def collapse_cols_to_str(df: pd.DataFrame) -> List[str]:
    """Collapses across columns in the dataframe to a single string.

    Args:
        df (pd.DataFrame): The dataframe to collapse.

    Returns:
        Tuple[str]: The collapsed dataframe.
    """
    all_columns = df.columns.tolist()
    cols_collapsed = [
        " ".join(x.split(","))
        for x in (df[all_columns].stack().groupby(level=0).apply(",".join))
    ]
    return cols_collapsed


def map_player_names(
    reference_df: pd.DataFrame, new_df: pd.DataFrame, *keys: str, match_threshold=80
) -> pd.DataFrame:
    """Maps player names between two dataframes using fuzzy matching.

    Args:
        reference_df (pd.DataFrame): Dataframe with reference player names,
            or names that other dataframes will be mapped to.
        new_df (pd.DataFrame): Dataframe with player names that will be
            mapped to the reference_df
        keys (str): Columns that will be used to match players
        match_threshold (int, optional): Threshold for fuzzy matching. Defaults to 80.

    Returns:
        pd.DataFrame: Dataframe with mapped player names

    Example:
        >>> ref_df = pd.DataFrame({"name": ["John Doe", "Jane Doe", "John Smith"]})
        >>> new_df = pd.DataFrame({"name": ["John Done", "Jane Roe", "John Smit"]})
        >>> map_player_names(ref_df, new_df, "name")

    """
    keys = list(keys)
    if "name" not in keys:
        raise ValueError("Must include 'name' in keys")
    reference_df = reference_df[keys].drop_duplicates()
    new_df = new_df[keys].drop_duplicates()
    new_df_match_lst = collapse_cols_to_str(new_df)
    new_df["is_match"] = 1
    reference_no_match_df = reference_df.merge(new_df, how="left", on=keys).query(
        "is_match != 1"
    )
    reference_no_match_lst = collapse_cols_to_str(reference_no_match_df)
    name_mapping_lst = []
    for row_index, correct_player_name in enumerate(reference_no_match_lst):
        name_proximity = [
            [index, fuzz.ratio(x, correct_player_name)]
            for index, x in enumerate(new_df_match_lst)
        ]
        match_index, match_proximity = sorted(
            name_proximity, key=lambda x: x[1], reverse=True
        )[0]
        if match_proximity > match_threshold:
            mapped_name_lst = new_df[keys].iloc[match_index].tolist() + [
                reference_no_match_df.iloc[row_index]["name"]
            ]
            name_mapping_lst.append(mapped_name_lst)
    name_mapping_df = pd.DataFrame(name_mapping_lst, columns=keys + ["mapped_name"])
    return name_mapping_df


def create_dir(dir_path: PosixPath) -> None:
    """Creates a directory if it does not exist.

    Args:
        dir_path (PosixPath): The path to the directory.

    Returns:
        None
    """
    if not Path.exists(dir_path):
        print("Creating directory:", dir_path)
        Path.mkdir(dir_path, parents=True)
    return None


def get_module_purpose(
    module_path: str, module_name_split_char: str = "_"
) -> Tuple[str, str]:
    """Extracts the the purpose of the module (collect or process) and
       data type that the module works with (e.g., stats, betting)
       from the module path.

    Args:
        module_path (str): The path to the module.
        module_name_split_char (str, optional): The character that seperate
            the module's purpose from the data type that the module works with.
            Defaults to "_".

    Raises:
        ValueError: If the character used to seperate the module's
            purpose from the data type
            is not present.
        ValueError: If the module has more than one character used to
                              seperate the module's purpose from the data type.
        ValueError: If the directory type is not an expected value.

    Returns:
        Tuple[str, str]: The module's purpose and data type.
    """
    dir_type_mapping = {
        "collect": "raw",
        "process": "processed",
        "validate": "validate",
    }
    expected_directories = dir_type_mapping.keys()
    module_name = module_path.split("/")[-1]
    if module_name_split_char not in module_name:
        raise ValueError(f"{module_name_split_char} not found in {module_name}")

    if len([x for x in list(module_name) if x == module_name_split_char]) != 1:
        raise ValueError(
            f"There must only be 1 '{module_name_split_char}' in {module_name}"
        )

    dir_type, data_type = module_name.split(module_name_split_char)
    if dir_type not in dir_type_mapping.keys():
        raise ValueError(f"Directory type must be one of {expected_directories}")

    data_type = data_type.replace(".py", "")
    dir_type = dir_type_mapping[dir_type]
    return (dir_type, data_type)


@pf.register_dataframe_method
def write_ff_csv(
    df: pd.DataFrame,
    root_dir: PosixPath,
    season_year: int,
    dir_type: str,
    data_type: str,
    *file_type: str,
) -> None:
    """Writes dataframe to a csv file, based on the season year, directory type,
        data type, and file type.

    Args:
        df (pd.DataFrame): The dataframe to write.
        root_dir (PosixPath): The root directory to write the dataframe to.
        season_year (int): The season year.
        dir_type (str): The directory type (e.g., raw, processed).
        data_type (str): The data type (e.g., stats, betting).
        file_type (str, optional): The file type (e.g., csv, json). Defaults to "csv".

    Raises:
        ValueError: If the file type is not an expected value.

    Returns:
        None
    """
    dir_types = ["raw", "processed"]
    if dir_type not in dir_types:
        raise ValueError(
            f"dir_type must be either 'raw' or 'processed', not {dir_type}"
        )
    dir_path = (
        root_dir
        / "staging_datasets"
        / "season"
        / str(season_year)
        / dir_type
        / data_type
    )
    create_dir(dir_path)
    fname = "_".join(file_type + (data_type,)) + ".csv"
    df.to_csv(dir_path / fname, index=False)
    return None


def concat_ff_csv(file_paths: List[PosixPath]) -> pd.DataFrame:
    """Concatenates the dataframes in the list of .csv file paths.

    Args:
        file_paths (List[PosixPath]): The list of .csv file paths.

    Returns:
        pd.DataFrame: The concatenated dataframe.
    """
    combined_df = pd.concat(
        [pd.read_csv(x, keep_default_na=False) for x in file_paths], ignore_index=True
    )
    return combined_df


def read_ff_csv(dir_path: PosixPath) -> pd.DataFrame:
    """Reads the .csv file(s) in the directory path. If there are multiple .csv files,
       concatenates them.

    Args:
        dir_path (PosixPath): The directory path containing the .csv file(s).

    Returns:
        pd.DataFrame: The concatenated dataframe if there are multiple .csv files,
            otherwise the dataframe.
    """
    file_paths = list(dir_path.glob("*.csv"))
    if not file_paths:
        raise FileNotFoundError(
            f"No .csv files found in {dir_path}. Have you run the collect module?"
        )
    if len(file_paths) > 1:
        df = concat_ff_csv(file_paths)
        return df
    else:
        df = pd.read_csv(file_paths[0], keep_default_na=False)
        return df


# TO DO: create logic to handle if function is called outside of the season
def fetch_current_week(calendar_df: pd.DataFrame) -> str:
    """Fetches the current week from the calendar dataframe."""
    # first determine which day of the week it is
    todays_date = datetime.today()
    todays_day_of_week = todays_date.strftime("%A")
    if todays_day_of_week == "Tuesday":
        sunday_date = todays_date + timedelta(days=5)
    elif todays_day_of_week == "Wednesday":
        sunday_date = todays_date + timedelta(days=4)
    elif todays_day_of_week == "Thursday":
        sunday_date = todays_date + timedelta(days=3)
    elif todays_day_of_week == "Friday":
        sunday_date = todays_date + timedelta(days=2)
    elif todays_day_of_week == "Saturday":
        sunday_date = todays_date + timedelta(days=1)
    # convert to datetime string
    sunday_date = sunday_date.strftime("%Y-%m-%d")
    # filter calendar to get the week
    week = str(calendar_df.query("date == @sunday_date")["week"].values[0])
    return week


def check_if_data_exists(
    new_data: pd.DataFrame, existing_data: pd.DataFrame, time_column: str
) -> bool:
    """For data that is updated as a weekly snapshot, check that is has not already been
    updated for the current time period (week). If it has,
    then do not append the new weekly data

    Args:
        new_data (pd.DataFrame): New data to be appended
        existing_data (pd.DataFrame): Existing data to be appended to
        time_column (str): Column name of the time period, such as week or date

    Returns:
        bool: True if the data has already been updated for the current time period,
        False otherwise

    Raises:
        ValueError: If the most_recent_time_value_existing_data
            is less than the most_recent_time_value_new_data
    """
    most_recent_time_value_new_data = max(new_data[time_column])
    most_recent_time_value_existing_data = max(existing_data[time_column])
    if most_recent_time_value_new_data == most_recent_time_value_existing_data:
        return True
    if most_recent_time_value_new_data > most_recent_time_value_existing_data:
        return False
    else:
        raise ValueError("New data should not be less than existing data")


def delete_files_in_staging_datasets_dir(dir_path: Path) -> None:
    """Deletes all files in the staging_datasets directory"""
    if "staging_datasets" not in str(dir_path):
        raise ValueError("deletion is restricted to staging data only")
    for file in dir_path.iterdir():
        file.unlink()


def dedup_with_agg(
    df: pd.DataFrame, keys: Union[str, list], agg_col: str, strategy: str
):
    """Deduplicate dataframe based on keys and aggregate agg_col based on strategy.

    Args:
        df (pd.DataFrame): dataframe to deduplicate
        keys (Union[str, list]): keys to group by
        agg_col (str): column to aggregate
        strategy (str): strategy to aggregate agg_col.
        Options are 'sum', 'mean', 'max', 'min'

    Returns:
        pd.DataFrame: deduplicated dataframe
    """
    # check if strategy is valid
    if strategy not in ["sum", "mean", "max", "min"]:
        raise ValueError(f"Invalid strategy: {strategy}")
    df = df.groupby(keys).agg({agg_col: strategy}).reset_index()
    return df
