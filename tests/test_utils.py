from pathlib import Path

import pandas as pd
import pytest

from fantasyfootball.errors import FantasyFootballError
from fantasyfootball.pipeline.utils import (
    concat_ff_csv,
    create_dir,
    get_module_purpose,
    retrieve_team_abbreviation,
)


@pytest.mark.parametrize(
    ("team_name", "expected"),
    (("Cleveland", "CLE"), ("Dallas Cowboys", "DAL"), ("Denver", "DEN")),
)
def test_retrieve_team_abbreviation(team_name, expected):
    assert retrieve_team_abbreviation(team_name) == expected


def test_retrieve_team_abbreviation_incorrect_name():
    with pytest.raises(ValueError):
        retrieve_team_abbreviation("Cincinnati Vikings")


@pytest.fixture(scope="module")
def create_test_dir():
    dir_path = Path(__file__).parent / "test_dir"
    yield dir_path
    Path.rmdir(dir_path)


def test_create_dir(create_test_dir):
    dir_path = create_test_dir
    create_dir(dir_path)
    assert Path.exists(dir_path)


@pytest.mark.parametrize(
    ("module_path", "expected"),
    (
        ("/path/to/collect/file/collect_stats.py", ("raw", "stats")),
        ("/path/to/process_betting.py", ("processed", "betting")),
    ),
)
def test_get_module_function(module_path, expected):
    assert get_module_purpose(module_path) == expected


def test_get_module_function_no_split_char_exception():
    with pytest.raises(FantasyFootballError):
        get_module_purpose("path/to/processing/processbetting.py")


def test_get_module_function_too_many_split_char_exception():
    with pytest.raises(FantasyFootballError):
        get_module_purpose("path/to/processing/process_betting_data.py")


def test_get_module_function_incorrect_data_type_exception():
    with pytest.raises(FantasyFootballError):
        get_module_purpose("path/to/processing/massage_stats.py")


@pytest.fixture(scope="module")
def create_csv_files():
    pd.DataFrame({"col1": ["a", "b"], "col2": [1, 2]}).to_csv("csv1.csv", index=False)
    pd.DataFrame({"col1": ["c", "d"], "col2": [3, 4]}).to_csv("csv2.csv", index=False)
    file_paths = list(Path.cwd().glob("*.csv"))
    yield file_paths
    for file_path in file_paths:
        Path.unlink(file_path)


def test_concat_ff_csv(create_csv_files):
    expected = pd.DataFrame({"col1": ["a", "b", "c", "d"], "col2": [1, 2, 3, 4]})
    file_paths = create_csv_files
    result = concat_ff_csv(file_paths)
    assert expected.equals(result)
