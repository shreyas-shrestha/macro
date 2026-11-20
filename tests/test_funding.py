import math
import socket

import numpy as np
import pandas as pd
import pytest

from macro.data import cache, series
from macro.data.calendar import INDEX_NAME, nyse_sessions
from macro.features import funding
from macro.features.funding import Funding
from macro.features.testing import assert_no_lookahead

SESSIONS = nyse_sessions("1997-01-02", "2017-12-29").rename(INDEX_NAME)
# The real series' missing sessions: its first two and 16 in the crisis, one gap of 9.
CRISIS = pd.to_datetime(
    ["2008-11-10", "2008-11-11", "2008-11-12", "2008-11-13", "2008-11-14"]
    + ["2008-12-12", "2008-12-15", "2008-12-16", "2008-12-17", "2008-12-18"]
    + ["2008-12-19", "2008-12-22", "2008-12-23", "2008-12-24", "2009-01-02", "2009-01-05"]
)
# A longer synthetic gap, so that later 252-session windows hold fewer than 200 values.
GAP_START = SESSIONS.get_loc(pd.Timestamp("2003-03-03"))
GAP = SESSIONS[GAP_START : GAP_START + 60]
MISSING = SESSIONS[:2].union(CRISIS).union(GAP)
# CPFF on sessions rises 0.01 pp a session, so session i holds i / 100.
CPFF = pd.Series(np.arange(len(SESSIONS)) / 100, index=SESSIONS).mask(SESSIONS.isin(MISSING))
# Sample s.d. of n consecutive integers.
SD_252 = math.sqrt(252 * 253 / 12)
SD_200 = math.sqrt(200 * 201 / 12)


@pytest.fixture(autouse=True)
def synthetic_cpff(monkeypatch):
    def load(name, end_date):
        assert name == "CPFF"
        return CPFF.loc[:end_date]

    monkeypatch.setattr(funding, "load", load)


def test_no_lookahead():
    cutoffs = ["2000-01-03", "2003-06-02", "2008-12-15", "2009-02-27", "2016-06-30"]
    assert_no_lookahead(Funding(), cutoffs=cutoffs)


@pytest.mark.skipif(
    not (cache.CACHE_DIR / "fred" / "CPFF.parquet").exists(), reason="no cached CPFF"
)
def test_no_lookahead_on_cached_cpff(monkeypatch):
    def no_network(*args):
        raise AssertionError("tests must not hit the network")

    # load reads an existing cache file whatever its length and fetches only without one.
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(funding, "load", series.load)
    cutoffs = ["2003-06-30", "2008-12-15", "2009-01-06", "2016-06-30"]
    assert_no_lookahead(Funding(), cutoffs=cutoffs)


def test_level_is_cpff_on_every_session_from_start_date():
    out = Funding().run("2017-12-31")
    assert list(out.columns) == ["cpff_level", "cpff_change_20d", "cpff_z_252"]
    pd.testing.assert_series_equal(
        out["cpff_level"], CPFF.loc[Funding.start_date :], check_names=False
    )


def test_first_row_uses_history_before_start_date():
    first = Funding().run("2017-12-31").iloc[0]
    assert first.name == Funding.start_date
    assert first["cpff_change_20d"] == pytest.approx(0.20)
    assert first["cpff_z_252"] == pytest.approx(125.5 / SD_252)


def test_change_is_missing_when_either_end_is():
    change = Funding().run("2017-12-31")["cpff_change_20d"]
    missing = SESSIONS.isin(MISSING)
    expected_nan = missing.copy()
    expected_nan[20:] |= missing[:-20]
    first = SESSIONS.get_loc(Funding.start_date)
    assert (change.isna().to_numpy() == expected_nan[first:]).all()
    assert change.dropna().to_numpy() == pytest.approx(0.20)
    # 20 sessions after 2008-11-10, across Thanksgiving: CPFF is back, its base is not.
    assert np.isnan(change["2008-12-09"]) and not np.isnan(CPFF["2008-12-09"])


def test_z_score_window_ends_at_t():
    z = Funding().run("2017-12-31")["cpff_z_252"]
    # 252 consecutive values, t the largest: t is 125.5 sessions above the mean.
    assert z["2005-06-30"] == pytest.approx(125.5 / SD_252)


def test_z_score_skips_missing_sessions_in_its_window():
    z = Funding().run("2017-12-31")["cpff_z_252"]
    t = SESSIONS.get_loc(pd.Timestamp("2009-03-31"))
    values = [i / 100 for i in range(t - 251, t + 1) if SESSIONS[i] not in MISSING]
    assert len(values) == 252 - 16
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    assert z["2009-03-31"] == pytest.approx((t / 100 - mean) / sd)


def test_z_score_needs_200_values():
    out = Funding().run("2017-12-31")
    # The window leaves the 60-session gap: 199 values, then 200, all after the gap.
    before, at = SESSIONS[GAP_START + 258], SESSIONS[GAP_START + 259]
    assert out.loc[[before, at], "cpff_level"].notna().all()
    assert np.isnan(out.loc[before, "cpff_z_252"])
    assert out.loc[at, "cpff_z_252"] == pytest.approx(99.5 / SD_200)
