import argparse
from pathlib import Path, PosixPath
from typing import List, Tuple

import pandas as pd
import pandas_flavor as pf

from fantasyfootball.errors import FantasyFootballError


def read_args() -> argparse.Namespace:
    """Helper function to read command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--season_year", type=int, help="The season year")
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
    team_abbreviation_mapping = {
        ("Arizona", "Arizona Cardinals"): "ARI",
        ("Atlanta", "Atlanta Falcons"): "ATL",
        ("Buffalo", "Buffalo Bills"): "BUF",
        ("Baltimore", "Baltimore Ravens"): "BAL",
        ("Carolina", "Carolina Panthers"): "CAR",
        ("Chicago", "Chicago Bears"): "CHI",
        ("Cincinnati", "Cincinnati Bengals"): "CIN",
        ("Cleveland", "Cleveland Browns"): "CLE",
        ("Dallas", "Dallas Cowboys"): "DAL",
        ("Denver", "Denver Broncos"): "DEN",
        ("Detroit", "Detroit Lions"): "DET",
        ("GreenBay", "Green Bay Packers"): "GNB",
        ("Houston", "Houston Texans"): "HOU",
        ("Indianapolis", "Indianapolis Colts"): "IND",
        ("Jacksonville", "Jacksonville Jaguars"): "JAX",
        ("KansasCity", "Kansas City Chiefs"): "KAN",
        ("LAChargers", "Los Angeles Chargers"): "LAC",
        ("Oakland", "Oakland Raiders"): "OAK",
        ("LARams", "Los Angeles Rams"): "LAR",
        ("LasVegas", "Las Vegas Raiders"): "LVR",
        ("Miami", "Miami Dolphins"): "MIA",
        ("Minnesota", "Minnesota Vikings"): "MIN",
        ("NewEngland", "New England Patriots"): "NWE",
        ("NewOrleans", "New Orleans Saints"): "NOR",
        ("NYGiants", "New York Giants"): "NYG",
        ("NYJets", "New York Jets"): "NYJ",
        ("Philadelphia", "Philadelphia Eagles"): "PHI",
        ("Pittsburgh", "Pittsburgh Steelers"): "PIT",
        ("SanFrancisco", "San Francisco 49ers"): "SFO",
        ("Seattle", "Seattle Seahawks"): "SEA",
        ("TampaBay", "Tampa Bay Buccaneers"): "TAM",
        ("Tennessee", "Tennessee Titans"): "TEN",
        ("Washington", "Washington Football Team"): "WAS",
    }
    for k, v in team_abbreviation_mapping.items():
        if team_name in k:
            return v
    raise ValueError(f"Team name {team_name} not found in team_abbreviation_mapping")


def create_dir(dir_path: PosixPath) -> None:
    """Creates a directory if it does not exist.

    Args:
        dir_path (PosixPath): The path to the directory.

    Returns:
        None
    """
    if not Path.exists(dir_path):
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
        FantasyFootballError: If the character used to seperate the module's
            purpose from the data type
            is not present.
        FantasyFootballError: If the module has more than one character used to
                              seperate the module's purpose from the data type.
        FantasyFootballError: If the directory type is not an expected value.

    Returns:
        Tuple[str, str]: The module's purpose and data type.
    """
    dir_type_mapping = {"collect": "raw", "process": "processed"}
    expected_directories = dir_type_mapping.keys()
    module_name = module_path.split("/")[-1]
    if module_name_split_char not in module_name:
        raise FantasyFootballError(
            f"{module_name_split_char} not found in {module_name}"
        )

    if len([x for x in list(module_name) if x == module_name_split_char]) != 1:
        raise FantasyFootballError(
            f"There must only be 1 '{module_name_split_char}' in {module_name}"
        )

    dir_type, data_type = module_name.split(module_name_split_char)
    if dir_type not in dir_type_mapping.keys():
        raise FantasyFootballError(
            f"Directory type must be one of {expected_directories}"
        )

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
        FantasyFootballError: If the file type is not an expected value.

    Returns:
        None
    """
    dir_types = ["raw", "processed"]
    if dir_type not in dir_types:
        raise FantasyFootballError(
            f"dir_type must be either 'raw' or 'processed', not {dir_type}"
        )
    dir_path = root_dir / "data" / "season" / str(season_year) / dir_type / data_type
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
    if len(file_paths) > 1:
        df = concat_ff_csv(file_paths)
        return df
    else:
        df = pd.read_csv(file_paths[0], keep_default_na=False)
        return df
