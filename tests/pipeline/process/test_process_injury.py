import numpy as np
import pandas as pd
import pytest

from fantasyfootball.pipeline.process.process_injury import (
    add_injury_feature_columns,
    _convert_plural_injury_to_singular,
    process_injury_type,
    add_missing_values_for_non_injured_players,
    fill_missing_values_for_non_injured_players,
)


def test_add_injury_feature_columns():
    pass


def test__convert_plural_injury_to_singular():
    pass


def test_process_injury_type():
    # injury_type = ["right hand", "left foot", "right shoulder",
    #           "Not Injury Related - Resting Player",
    # "Knee,not Injury Related - Resting Player",
    #           "Finger/right Hand", "Ankle,quadricep",
    #  "Hamstring,ankle", "load management, knee",
    #           "achilles", "rib", "ribs",
    # "hands", "hand", "", "core", "abdomen"
    #           ]
    pass


def test_add_missing_values_for_non_injured_players():
    pass


def test_fill_missing_values_for_non_injured_players():
    pass
