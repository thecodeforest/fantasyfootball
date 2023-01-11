import sys
from pathlib import Path

import pytest

sys.path.append(str(Path.cwd()))

from pipeline.collect.collect_stats import (  # noqa: E402
    clean_player_name,
    create_abbr_name_combo,
    create_player_id,
    first_name_is_abbr,
)


@pytest.mark.parametrize(
    ("first_name", "expected"),
    (
        ("D.J. Green", True),
        ("Tom Brady", False),
        ("D.K. Metcalf", True),
        ("DJ Chark", False),
        ("D'Wayne Eskridge", False),
        ("Amon-Ra St. Brown", False),
    ),
)
def test_first_name_is_abbr(first_name, expected):
    assert first_name_is_abbr(first_name) == expected


@pytest.mark.parametrize(
    ("name", "_type", "expected"),
    (
        ("Tom", "first", "Tom"),
        ("Brady", "last", "Brady"),
        ("DJ", "first", "DJ"),
        ("Chark", "last", "Chark"),
        ("D'Wayne", "first", "DWayne"),
        ("Peoples-Jones", "last", "PeoplesJones"),
    ),
)
def test_clean_name(name, _type, expected):
    assert clean_player_name(name, _type) == expected


def test_valid_clean_name():
    with pytest.raises(ValueError):
        clean_player_name("John", "middle")


def test_create_abbr_name_combo():
    first_name = "D.J."
    expected = ("DJ", "D.", "D")
    assert create_abbr_name_combo(first_name) == expected


def test_create_player_id_no_abbr():
    first_name = "Tom"
    last_name = "Brady"
    expected = [
        "BradTo00",
        "BradTo01",
        "BradTo02",
        "BradTo03",
        "BradTo04",
        "BradTo05",
        "BradTo06",
        "BradTo07",
        "BradTo08",
        "BradTo09",
    ]
    assert create_player_id(first_name, last_name) == expected


def test_create_player_id_abbr():
    first_name = "D.K."
    last_name = "Metcalf"
    expected = [
        "MetcDK00",
        "MetcD.00",
        "MetcD00",
        "MetcDK01",
        "MetcD.01",
        "MetcD01",
        "MetcDK02",
        "MetcD.02",
        "MetcD02",
        "MetcDK03",
        "MetcD.03",
        "MetcD03",
        "MetcDK04",
        "MetcD.04",
        "MetcD04",
        "MetcDK05",
        "MetcD.05",
        "MetcD05",
        "MetcDK06",
        "MetcD.06",
        "MetcD06",
        "MetcDK07",
        "MetcD.07",
        "MetcD07",
        "MetcDK08",
        "MetcD.08",
        "MetcD08",
        "MetcDK09",
        "MetcD.09",
        "MetcD09",
    ]
    assert create_player_id(first_name, last_name) == expected


def test_create_player_id_apostrophe():
    first_name = "James"
    last_name = "O'Shaughnessys"
    expected = [
        "O'ShJa00",
        "OShaJa00",
        "O'ShJa01",
        "OShaJa01",
        "O'ShJa02",
        "OShaJa02",
        "O'ShJa03",
        "OShaJa03",
        "O'ShJa04",
        "OShaJa04",
        "O'ShJa05",
        "OShaJa05",
        "O'ShJa06",
        "OShaJa06",
        "O'ShJa07",
        "OShaJa07",
        "O'ShJa08",
        "OShaJa08",
        "O'ShJa09",
        "OShaJa09",
    ]
    assert create_player_id(first_name, last_name) == expected
