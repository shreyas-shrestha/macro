import socket

import numpy as np
import pandas as pd
import pytest

from macro.data import alfred, cache, claims, revised, series
from macro.features.slow_macro import (
    Claims,
    FinancialConditions,
    claims_features,
    conditions_features,
)
from macro.features.testing import assert_no_lookahead

# Weeks ending Saturday, as FRED dates ICNSA; one value per week, encoding its date.
SATURDAYS = pd.date_range("1988-01-02", "2017-12-30", freq="W-SAT")
ICNSA = pd.Series(300_000 + 100.0 * np.arange(len(SATURDAYS)), index=SATURDAYS)
ICNSA += 50_000 * np.sin(2 * np.pi * SATURDAYS.day_of_year / 365.25)
# ALFRED first releases from the week ending 2009-05-30, 1% below the vintage, each
# released on Thursday s + 5.
FIRST_WEEKS = SATURDAYS[SATURDAYS >= "2009-05-30"]
FIRST = 0.99 * ICNSA.loc[FIRST_WEEKS]
FIRST_ROWS = [
    {"date": f"{d:%Y-%m-%d}", "realtime_start": f"{d + pd.Timedelta(days=5):%Y-%m-%d}", "value": v}
    for d, v in FIRST.items()
]
SPLICED = FIRST.combine_first(ICNSA)

# ANFCI first releases: the week ending Friday f is released the next Thursday f + 6,
# except that the weeks of 2012-10-26 and 2012-11-02 were both released on 2012-11-07.
FRIDAYS = pd.date_range("2011-05-27", "2017-12-22", freq="W-FRI")
RELEASED = FRIDAYS + pd.Timedelta(days=6)
LATE = FRIDAYS.isin(pd.to_datetime(["2012-10-26", "2012-11-02"]))
RELEASED = RELEASED.where(~LATE, pd.Timestamp("2012-11-07"))
ANFCI = pd.Series(np.arange(len(FRIDAYS)) / 100, index=RELEASED)


@pytest.fixture(autouse=True)
def synthetic_sources(tmp_path, monkeypatch):
    def no_network(*args):
        raise AssertionError("tests must not hit the network")

    class FakeFred:
        def __init__(self, api_key):
            pass

        def get_series(self, series_id, observation_start, observation_end, **kwargs):
            return ICNSA.loc[observation_start:observation_end]

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("FRED_API_KEY", "test")
    monkeypatch.setattr(claims, "Fred", FakeFred)
    monkeypatch.setattr(alfred, "first_releases", lambda sid, end: alfred.table(FIRST_ROWS))
    monkeypatch.setitem(series.FETCHERS, "alfred", lambda sid, start, end: ANFCI.loc[start:end])


def four_week_average(week_end):
    return SPLICED.loc[week_end - pd.Timedelta(weeks=3) : week_end].mean()


def test_claims_no_lookahead():
    cutoffs = ["1990-06-29", "2008-09-15", "2011-12-30", "2017-11-24"]
    assert_no_lookahead(Claims(), cutoffs=cutoffs)


def test_financial_conditions_no_lookahead():
    cutoffs = ["2011-06-30", "2012-11-07", "2012-11-08", "2015-03-31"]
    assert_no_lookahead(FinancialConditions(), cutoffs=cutoffs)


def test_claims_features_by_hand():
    thursdays = pd.date_range("2000-01-06", periods=60, freq="W-THU")
    weekly = pd.Series(10.0, index=thursdays)
    weekly.iloc[4:8] = 5.0  # the 4-week average a year before the last week is 5
    weekly.iloc[20:24] = 2.0  # the lowest 4-week average in the prior 52 weeks is 2
    weekly.iloc[56:60] = 20.0
    out = claims_features(weekly)
    assert out["claims_yoy"].iloc[59] == pytest.approx(np.log(20 / 5))
    assert out["claims_vs_low"].iloc[59] == pytest.approx(np.log(20 / 2))
    # The first full year of 4-week averages ends at week 54; week 54 lacks one.
    assert out.iloc[54].isna().all() and out.iloc[55].notna().all()
    # A missing week leaves the year-ago match by date intact.
    gapped = claims_features(weekly.drop(thursdays[30]))
    assert gapped["claims_yoy"].iloc[-1] == pytest.approx(np.log(20 / 5))


def test_claims_on_sessions_follow_the_release_thursday():
    out = Claims().run("2017-12-31")
    # Week ending 2017-11-11, released Thursday 11-16, first used Friday 11-17.
    week = pd.Timestamp("2017-11-11")
    expected = np.log(four_week_average(week) / four_week_average(week - pd.Timedelta(weeks=52)))
    assert out.loc["2017-11-16", "claims_yoy"] != pytest.approx(expected)
    assert out.loc["2017-11-17", "claims_yoy"] == pytest.approx(expected)
    # Thanksgiving: week ending 11-18 is stamped Thursday 11-23 and first used Monday 11-27.
    assert out.loc["2017-11-24", "claims_yoy"] == out.loc["2017-11-17", "claims_yoy"]
    assert out.loc["2017-11-27", "claims_yoy"] != out.loc["2017-11-24", "claims_yoy"]


def test_claims_read_first_releases_from_mid_2009_and_the_vintage_before():
    weekly = series.raw("ICNSA")
    thursday = pd.Timedelta(days=5)
    assert weekly[pd.Timestamp("2009-05-23") + thursday] == ICNSA["2009-05-23"]
    assert weekly[pd.Timestamp("2009-05-30") + thursday] == FIRST["2009-05-30"]
    # The year-ago average of the week ending 2009-12-05 is all vintage values.
    out = Claims().run("2017-12-31")
    week = pd.Timestamp("2009-12-05")
    expected = np.log(four_week_average(week) / four_week_average(week - pd.Timedelta(weeks=52)))
    assert out.loc["2009-12-11", "claims_yoy"] == pytest.approx(expected)
    assert four_week_average(week) == pytest.approx(FIRST.loc[:week].iloc[-4:].mean())


def test_claims_year_ago_is_52_weeks_back_across_a_53_week_year():
    # 2011 has 53 Saturdays; the week ending 2011-12-31 is compared with 2011-01-01.
    out = Claims().run("2017-12-31")
    week = pd.Timestamp("2011-12-31")
    expected = np.log(four_week_average(week) / four_week_average(pd.Timestamp("2011-01-01")))
    assert out.loc["2012-01-06", "claims_yoy"] == pytest.approx(expected)


def test_claims_vs_low_uses_the_prior_52_weeks_only():
    out = Claims().run("2017-12-31")
    week = pd.Timestamp("2014-06-07")
    prior = [four_week_average(week - pd.Timedelta(weeks=k)) for k in range(1, 53)]
    expected = np.log(four_week_average(week) / min(prior))
    assert out.loc["2014-06-13", "claims_vs_low"] == pytest.approx(expected)


def test_conditions_features_by_hand():
    weekly = pd.Series(np.arange(20) / 10, index=pd.date_range("2012-01-05", periods=20, freq="W"))
    out = conditions_features(weekly)
    assert out["anfci_change_13w"].iloc[:13].isna().all()
    assert out["anfci_change_13w"].iloc[13:].to_numpy() == pytest.approx(1.3)
    pd.testing.assert_series_equal(out["anfci_level"], weekly, check_names=False)


def test_financial_conditions_use_first_releases_in_publication_order():
    out = FinancialConditions().run("2017-12-31")
    assert out.index[0] == pd.Timestamp("2011-06-03")
    assert out.loc["2011-06-03", "anfci_level"] == 0.0
    later = FRIDAYS.get_loc(pd.Timestamp("2012-11-02"))
    assert out.loc["2012-11-07", "anfci_level"] == pytest.approx((later - 2) / 100)
    assert out.loc["2012-11-08", "anfci_level"] == pytest.approx(later / 100)
    assert out.loc["2012-11-08", "anfci_change_13w"] == pytest.approx(0.13)


# The revised vintage: every Friday since 1990, one value per week encoding its date.
VINTAGE_FRIDAYS = pd.date_range("1990-01-05", "2017-12-29", freq="W-FRI")
VINTAGE = pd.Series(np.arange(len(VINTAGE_FRIDAYS)) / 1000, index=VINTAGE_FRIDAYS)


@pytest.fixture
def revised_vintage(monkeypatch):
    requests = []

    class FakeFred:
        def __init__(self, api_key):
            pass

        def get_series(self, series_id, observation_start, observation_end, **kwargs):
            requests.append(kwargs)
            return VINTAGE.loc[observation_start:observation_end]

    monkeypatch.setattr(revised, "Fred", FakeFred)
    return requests


def test_revised_weeks_are_stamped_the_thursday_after_the_week(revised_vintage):
    out = revised.fetch("ANFCI", pd.Timestamp("1990-01-02"), pd.Timestamp("2017-12-31"))
    assert revised_vintage == [{"realtime_start": "2017-12-31", "realtime_end": "2017-12-31"}]
    assert out.index[0] == pd.Timestamp("1990-01-11")
    assert (out.index.day_of_week == 3).all()
    # The week ending 2017-12-29 is stamped 2018-01-04 and dropped.
    assert out.index[-1] == pd.Timestamp("2017-12-28")


def test_revised_refuses_weeks_not_ending_friday(monkeypatch):
    class FakeFred:
        def __init__(self, api_key):
            pass

        def get_series(self, *args, **kwargs):
            return pd.Series([1.0], index=pd.to_datetime(["2017-12-28"]))

    monkeypatch.setattr(revised, "Fred", FakeFred)
    with pytest.raises(ValueError, match="Friday"):
        revised.fetch("ANFCI", pd.Timestamp("2017-01-02"), pd.Timestamp("2017-12-31"))


def test_revised_financial_conditions_start_in_1990_and_lag_the_release(revised_vintage):
    out = FinancialConditions(revised=True).run("2017-12-31")
    assert out.index[0] == pd.Timestamp("1990-01-02")
    # The week ending Friday 2008-09-12 is stamped Thursday 09-18 and first used Friday 09-19.
    week = VINTAGE_FRIDAYS.get_loc(pd.Timestamp("2008-09-12"))
    assert out.loc["2008-09-18", "anfci_level"] == pytest.approx((week - 1) / 1000)
    assert out.loc["2008-09-19", "anfci_level"] == pytest.approx(week / 1000)
    assert out.loc["2008-09-19", "anfci_change_13w"] == pytest.approx(0.013)


def test_revised_financial_conditions_no_lookahead(revised_vintage):
    cutoffs = ["1995-06-30", "2008-09-18", "2008-09-19", "2015-03-31"]
    assert_no_lookahead(FinancialConditions(revised=True), cutoffs=cutoffs)
