import pandas as pd
import pytest

from fantasyfootball.pipeline.process.process_betting import (
    clean_game_date,
    create_game_id,
    create_point_spread_df
)


@pytest.mark.parametrize(
    ("season_year", "date", "expected"),
    (
        (2020, "1003", "2020-10-03"),
        (2021, "1212", "2021-12-12"),
        (2019, "926", "2019-09-26"),
    ),
)
def test_clean_game_date(season_year, date, expected):
    assert clean_game_date(season_year, date) == expected


def test_create_game_id():
    df = pd.DataFrame({"data": list(range(1, 9))})
    expected = pd.concat(
        [df.copy(), pd.DataFrame({"game_id": [1, 1, 2, 2, 3, 3, 4, 4]})], axis=1
    )
    result = df.create_game_id()
    assert expected.equals(result)


def test_create_point_spread_df():
    df = pd.DataFrame(
        [["DAL", "52.5", 375, 1], ["TAM", "7", -450, 1]],
        columns=["team", "open", "ml", "game_id"],
    )
    expected = pd.DataFrame(
        [["TAM", "DAL", 30], ["DAL", "TAM", 23]],
        columns=["team", "opp", "projected_off_pts"],
    )
    result = df.create_point_spread_df()
    assert expected.equals(result)


def test_create_point_spread_df_even_moneyline():
    df = pd.DataFrame(
        [["LAC", "45", 110, 6], ["WAS", "pk", -130, 6]],
        columns=["team", "open", "ml", "game_id"],
    )
    expected = pd.DataFrame(
        [["LAC", "WAS", 22], ["WAS", "LAC", 22]],
        columns=["team", "opp", "projected_off_pts"],
    )
    result = df.create_point_spread_df()
    assert expected.equals(result)
