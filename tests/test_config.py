from nba_predictor.config import (
    BASE_URL,
    RANDOM_SEED,
    SEASON_END,
    SEASON_START,
    TRAIN_CUTOFF,
)


def test_constants_have_expected_types():
    assert isinstance(BASE_URL, str)
    assert isinstance(RANDOM_SEED, int)
    assert isinstance(SEASON_START, int)
    assert isinstance(SEASON_END, int)
    assert isinstance(TRAIN_CUTOFF, int)


def test_season_range_is_valid():
    assert SEASON_START < SEASON_END
    assert TRAIN_CUTOFF >= SEASON_START
    assert TRAIN_CUTOFF <= SEASON_END
