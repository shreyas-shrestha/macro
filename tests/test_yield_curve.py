import numpy as np
import pandas as pd
import pytest

from macro.config import DEV_END
from macro.data.calendar import nyse_sessions
from macro.data.pit import align
from macro.features.base import INDEX_NAME
from macro.features.testing import assert_no_lookahead
from macro.features.yield_curve import COLUMNS, YieldCurve

HISTORY_START = pd.Timestamp("1998-01-02")
CUTOFFS = ["2000-06-30", "2005-10-11", "2010-12-31"]


def synthetic_raw(seed: int, start_level: float) -> pd.Series:
    """A random walk stamped on weekdays, with a few missing observations."""
    days = pd.bdate_range(HISTORY_START, DEV_END)
    rng = np.random.default_rng(seed)
    values = start_level + rng.normal(0.0, 0.05, len(days)).cumsum()
    values[rng.choice(len(days), 40, replace=False)] = np.nan
    return pd.Series(values, index=days)


RAW = {"T10Y2Y": synthetic_raw(1, 1.0), "DGS10": synthetic_raw(2, 5.0)}


def loader(name: str, end_date: pd.Timestamp) -> pd.Series:
    """Stand-in for ``macro.data.series.load``: same alignment, no network."""
    sessions = nyse_sessions(HISTORY_START, end_date).rename(INDEX_NAME)
    return align(RAW[name].loc[:end_date], sessions, 1, 3).rename(name)


@pytest.fixture(scope="module")
def full() -> pd.DataFrame:
    return YieldCurve(loader).run(DEV_END)


def test_no_lookahead():
    assert_no_lookahead(YieldCurve(loader), cutoffs=CUTOFFS)


def test_columns_and_range(full):
    assert tuple(full.columns) == COLUMNS
    assert full.index[0] == pd.Timestamp("2000-01-03")
    assert full.index[-1] <= DEV_END
    assert (full.dtypes == np.float64).all()


def test_windows_are_full_at_start(full):
    assert full.iloc[0].notna().all()


def test_features_match_definitions(full):
    slope = loader("T10Y2Y", DEV_END)
    ten_year = loader("DGS10", DEV_END)
    t = pd.Timestamp("2010-06-15")
    pos = slope.index.get_loc(t)

    assert full.at[t, "t10y2y_level"] == slope.iloc[pos]
    assert full.at[t, "t10y2y_change_20"] == pytest.approx(slope.iloc[pos] - slope.iloc[pos - 20])

    window = slope.iloc[pos - 252 : pos]  # the 252 sessions before t, excluding t
    expected_z = (slope.iloc[pos] - window.mean()) / window.std()
    assert full.at[t, "t10y2y_z_252"] == pytest.approx(expected_z)

    changes = ten_year.diff().iloc[pos - 19 : pos + 1]
    assert full.at[t, "dgs10_vol_20"] == pytest.approx(changes.std())


def test_z_score_ignores_todays_value_in_its_window():
    """A spike at t moves the numerator at t, never the mean or std used at t."""
    t = pd.Timestamp("2010-06-15")
    base = YieldCurve(loader).run(t)

    def spiked(name, end_date):
        series = loader(name, end_date).copy()
        if name == "T10Y2Y":
            series.loc[t] += 1.0
        return series

    slope = loader("T10Y2Y", t)
    std = slope.iloc[-253:-1].std()
    bumped = YieldCurve(spiked).run(t)
    assert bumped.at[t, "t10y2y_z_252"] - base.at[t, "t10y2y_z_252"] == pytest.approx(1.0 / std)


def test_empty_before_start():
    frame = YieldCurve(loader).run("1999-12-31")
    assert frame.empty
    assert tuple(frame.columns) == COLUMNS


def test_holdout_is_refused():
    with pytest.raises(ValueError, match="holdout"):
        YieldCurve().run("2018-01-02")
