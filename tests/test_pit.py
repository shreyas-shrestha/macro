import numpy as np
import pandas as pd
import pytest

from macro.data.calendar import nyse_sessions
from macro.data.pit import align
from macro.data.series import SERIES

SESSIONS = nyse_sessions("2001-01-02", "2001-06-29")


def test_value_enters_lag_sessions_after_stamp():
    raw = pd.Series([1.0], index=[SESSIONS[10]])
    aligned = align(raw, SESSIONS, lag=2, max_carry=0)
    assert aligned.iloc[12] == 1.0
    assert aligned.drop(SESSIONS[12]).isna().all()


def test_good_friday_value_is_not_early():
    # 2001-04-13 is Good Friday: a business day but not an NYSE session.
    sessions = nyse_sessions("2001-04-09", "2001-04-20")
    dates = pd.to_datetime(["2001-04-12", "2001-04-13", "2001-04-16"])
    raw = pd.Series([1.0, 2.0, 3.0], index=dates)
    aligned = align(raw, sessions, lag=2, max_carry=0)
    # Thu 12th -> Tue 17th. Fri 13th and Mon 16th both map to Wed 18th; the latest wins.
    assert aligned[pd.Timestamp("2001-04-17")] == 1.0
    assert aligned[pd.Timestamp("2001-04-18")] == 3.0


def test_equal_stamps_keep_the_last_row():
    # A series' first vintage stamps its whole back history with one date.
    raw = pd.Series(np.arange(50, dtype=float), index=[SESSIONS[3]] * 50)
    assert align(raw, SESSIONS, lag=0, max_carry=0).iloc[3] == 49.0


def test_missing_value_does_not_displace_a_real_one():
    # Good Friday's value and Monday's NaN both map to Monday; the NaN must not win.
    sessions = nyse_sessions("2001-04-09", "2001-04-20")
    raw = pd.Series([2.0, np.nan], index=pd.to_datetime(["2001-04-13", "2001-04-16"]))
    assert align(raw, sessions, lag=0, max_carry=0)[pd.Timestamp("2001-04-16")] == 2.0


def test_carry_stops_at_limit():
    raw = pd.Series(np.arange(len(SESSIONS), dtype=float), index=SESSIONS)
    raw.iloc[20:30] = np.nan
    aligned = align(raw, SESSIONS, lag=0, max_carry=3)
    assert (aligned.iloc[20:23] == raw.iloc[19]).all()
    assert aligned.iloc[23:30].isna().all()


def test_values_past_last_session_are_dropped():
    raw = pd.Series([1.0], index=[SESSIONS[-1]])
    assert align(raw, SESSIONS, lag=1, max_carry=0).isna().all()


@pytest.mark.parametrize("lag, max_carry", sorted({(s.lag, s.max_carry) for s in SERIES.values()}))
def test_truncating_raw_after_t_never_changes_values_up_to_t(lag, max_carry):
    # Daily stamps on business days (including NYSE holidays) with random gaps.
    rng = np.random.default_rng(8)
    days = pd.bdate_range(SESSIONS[0], SESSIONS[-1])
    raw = pd.Series(rng.normal(size=len(days)), index=days)
    raw[rng.random(len(days)) < 0.1] = np.nan
    full = align(raw, SESSIONS, lag, max_carry)
    for t in SESSIONS[::7]:
        sessions_t = SESSIONS[SESSIONS <= t]
        truncated = align(raw.loc[:t], sessions_t, lag, max_carry)
        pd.testing.assert_series_equal(truncated, full.loc[:t])
