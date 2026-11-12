import numpy as np
import pandas as pd
import pytest

from macro.data.calendar import nyse_sessions
from macro.features.credit import Credit, align
from macro.features.testing import assert_no_lookahead

END = pd.Timestamp("2003-12-31")
SESSIONS = nyse_sessions(Credit.history_start, END)
# FRED dates its daily series on US business days, not NYSE sessions.
BDAYS = pd.bdate_range(Credit.history_start, END)
RNG = np.random.default_rng(11)
RAW = {
    "BAA10Y": pd.Series(2.0 + np.cumsum(RNG.normal(0, 0.02, len(BDAYS))), index=BDAYS),
    "DAAA": pd.Series(6.0 + np.cumsum(RNG.normal(0, 0.01, len(BDAYS))), index=BDAYS),
}
RAW["DBAA"] = RAW["DAAA"] + 0.8 + np.cumsum(RNG.normal(0, 0.005, len(BDAYS)))


def synthetic_loader(raw=RAW):
    def load(series_id, start, end):
        return raw[series_id].loc[start:end]

    return load


def test_no_lookahead():
    signal = Credit(loader=synthetic_loader())
    assert_no_lookahead(signal, cutoffs=["2000-06-30", "2001-09-17", "2002-12-31"], end_date=END)


def test_value_enters_two_sessions_after_observation():
    out = Credit(loader=synthetic_loader()).run(END)
    s = pd.Timestamp("2001-03-05")  # Monday, a normal session
    pos = SESSIONS.get_loc(s)
    assert out.loc[SESSIONS[pos + 2], "baa10y_level"] == RAW["BAA10Y"][s]
    assert out.loc[SESSIONS[pos + 1], "baa10y_level"] != RAW["BAA10Y"][s]
    assert out.loc[s, "baa10y_level"] != RAW["BAA10Y"][s]


def test_good_friday_observation_is_not_early():
    # 2001-04-13 is Good Friday: a business day but not an NYSE session.
    sessions = nyse_sessions("2001-04-09", "2001-04-20")
    dates = pd.to_datetime(["2001-04-12", "2001-04-13", "2001-04-16"])
    raw = pd.Series([1.0, 2.0, 3.0], index=dates)
    aligned = align(raw, sessions, lag=2)
    assert pd.Timestamp("2001-04-13") not in aligned.index
    # Thu 12th -> usable Tue 17th; Fri 13th and Mon 16th both map to Wed 18th, latest wins.
    assert aligned[pd.Timestamp("2001-04-17")] == 1.0
    assert aligned[pd.Timestamp("2001-04-18")] == 3.0


def test_bond_holiday_gap_is_carried_but_long_gaps_are_not():
    sessions = nyse_sessions("2001-01-02", "2001-02-28")
    raw = pd.Series(np.arange(len(sessions), dtype=float), index=sessions)
    raw.iloc[10] = np.nan  # one-day bond holiday
    raw.iloc[20:30] = np.nan  # long outage
    aligned = align(raw, sessions, lag=2)
    assert aligned.iloc[12] == raw.iloc[9]
    # raw[19] is usable at session 21 and carried for three more sessions.
    assert aligned.iloc[24] == raw.iloc[19]
    assert np.isnan(aligned.iloc[25])


def test_change_and_zscore_use_only_past_values():
    out = Credit(loader=synthetic_loader()).run(END)
    level = out["baa10y_level"]
    full = Credit(loader=synthetic_loader())
    t = pd.Timestamp("2002-06-03")
    pos = out.index.get_loc(t)
    assert out.loc[t, "baa10y_change_20"] == pytest.approx(level.iloc[pos] - level.iloc[pos - 20])
    # Rebuild the z-score by hand from the aligned series including pre-start history.
    sessions = nyse_sessions(full.history_start, END)
    aligned = align(RAW["BAA10Y"].loc[: sessions[-3]], sessions, 2)
    window = aligned.loc[:t].iloc[-253:-1]
    expected = (aligned[t] - window.mean()) / window.std()
    assert out.loc[t, "baa10y_z_252"] == pytest.approx(expected)


def test_quality_spread_is_baa_minus_aaa():
    out = Credit(loader=synthetic_loader()).run(END)
    t = pd.Timestamp("2002-06-03")
    s = SESSIONS[SESSIONS.get_loc(t) - 2]
    assert out.loc[t, "baa_aaa_level"] == pytest.approx(RAW["DBAA"][s] - RAW["DAAA"][s])


def test_features_start_populated_on_start_date():
    out = Credit(loader=synthetic_loader()).run(END)
    assert out.index[0] == Credit.start_date
    assert out.iloc[0].notna().all()


def test_holdout_is_locked():
    with pytest.raises(ValueError, match="holdout"):
        Credit(loader=synthetic_loader()).run("2018-01-02")


def test_loader_never_asked_past_last_usable_observation():
    requested = []

    def spy(series_id, start, end):
        requested.append(end)
        return RAW[series_id].loc[start:end]

    Credit(loader=spy).run("2002-12-31")
    sessions = nyse_sessions(Credit.history_start, "2002-12-31")
    assert max(requested) == sessions[-3]
