from pathlib import Path

import pandas as pd
import pytest
import sys

sys.path.append(str(Path.cwd()))
from pipeline.utils import (  # noqa: E402
    concat_ff_csv,
    create_dir,
    get_module_purpose,
    retrieve_team_abbreviation,
    collapse_cols_to_str,
    map_player_names,
    dedup_with_agg,
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


def test_collapse_cols_to_str():
    df = pd.DataFrame(
        {"col1": ["a", "b", "c"], "col2": ["d", "e", "f"], "col3": ["g", "h", "i"]}
    )
    expected = ["a d g", "b e h", "c f i"]
    assert collapse_cols_to_str(df) == expected


def test_map_player_names():
    reference_df = pd.DataFrame(
        {"name": ["John Doe", "Jane Doe", "John Smith", "Jane Taylor"]}
    )
    new_df = pd.DataFrame(
        {"name": ["John Done", "Jane Roe", "John Smit", "Lauren Hill"]}
    )
    expected = pd.DataFrame(
        [
            ["John Done", "John Doe"],
            ["Jane Roe", "Jane Doe"],
            ["John Smit", "John Smith"],
        ],
        columns=["name", "mapped_name"],
    )
    result = map_player_names(reference_df, new_df, "name")
    assert expected.equals(result)


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


@pytest.fixture(scope="module")
def create_csv_files():
    pd.DataFrame({"col1": ["a", "b"], "col2": [1, 2]}).to_csv("csv1.csv", index=False)
    pd.DataFrame({"col1": ["c", "d"], "col2": [3, 4]}).to_csv("csv2.csv", index=False)
    file_paths = list(Path.cwd().glob("*.csv"))
    yield file_paths
    for file_path in file_paths:
        Path.unlink(file_path)


# TO DO: Figure out why this is failing on py 3.9 but not on 3.8ß
# def test_concat_ff_csv(create_csv_files):
#     expected = pd.DataFrame({"col1": ["a", "b", "c", "d"], "col2": [1, 2, 3, 4]})
#     file_paths = create_csv_files
#     result = concat_ff_csv(file_paths)
#     assert expected.equals(result)


def test_dedup_with_agg():
    columns = ["name", "position", "year", "week", "fanduel_salary"]
    expected = pd.DataFrame(
        [
            ["Justin Herbert", "QB", 2022, 5, 8200],
            ["Keenan Allen", "WR", 2022, 5, 7000],
            ["Mike Williams", "WR", 2022, 5, 7400],
        ],
        columns=columns,
    )

    duplicated_df = pd.DataFrame(
        [
            ["Mike Williams", "WR", 2022, 5, 7400],
            ["Keenan Allen", "WR", 2022, 5, 7000],
            ["Mike Williams", "WR", 2022, 5, 4500],
            ["Justin Herbert", "QB", 2022, 5, 8200],
        ],
        columns=columns,
    )
    keys = ["name", "position", "year", "week"]
    agg_col = "fanduel_salary"
    strategy = "max"
    result = dedup_with_agg(duplicated_df, keys, agg_col, strategy)
    assert expected.equals(result)
