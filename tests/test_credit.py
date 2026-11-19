import numpy as np
import pandas as pd
import pytest

from macro.data import cache, fred, series
from macro.data.calendar import nyse_sessions
from macro.features.credit import BACKFILL_SESSIONS, Credit
from macro.features.testing import assert_no_lookahead

END = pd.Timestamp("2017-12-29")
# FRED dates observations on US business days; fred.fetch stamps them by posting day.
OBS = pd.bdate_range("1990-01-02", END)
RNG = np.random.default_rng(11)
RAW = {
    "BAA10Y": pd.Series(2.0 + np.cumsum(RNG.normal(0, 0.02, len(OBS))), index=OBS),
    "DAAA": pd.Series(6.0 + np.cumsum(RNG.normal(0, 0.01, len(OBS))), index=OBS),
}
RAW["DBAA"] = RAW["DAAA"] + 0.8 + np.cumsum(RNG.normal(0, 0.005, len(OBS)))


@pytest.fixture(autouse=True)
def fake_fred(tmp_path, monkeypatch):
    """Serve RAW through the real ``load``: stamped by posting day, cached in tmp."""

    def fetch(series_id, start, end):
        observed = RAW[series_id].loc[start:end]
        posted = pd.Series(observed.to_numpy(), index=fred.posting_days(observed.index))
        return posted.loc[start:end]

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    monkeypatch.setitem(series.FETCHERS, "fred", fetch)


def test_no_lookahead():
    assert_no_lookahead(Credit(), cutoffs=["2002-06-28", "2008-09-15", "2014-12-31"])


def test_value_observed_on_s_enters_two_sessions_later():
    out = Credit().run(END)
    sessions = nyse_sessions("2000-01-03", END)
    s = pd.Timestamp("2005-03-07")  # Monday: posted Tuesday, used Wednesday
    pos = sessions.get_loc(s)
    assert out.loc[sessions[pos + 2], "baa10y_level"] == RAW["BAA10Y"][s]
    assert out.loc[sessions[pos + 1], "baa10y_level"] != RAW["BAA10Y"][s]


def test_change_and_zscore_use_only_past_values():
    out = Credit().run(END)
    level = series.load("BAA10Y", END)
    t = pd.Timestamp("2009-06-01")
    pos = level.index.get_loc(t)
    assert out.loc[t, "baa10y_change_20"] == pytest.approx(level.iloc[pos] - level.iloc[pos - 20])
    window = level.iloc[pos - 252 : pos]
    assert out.loc[t, "baa10y_z_252"] == pytest.approx(
        (level.iloc[pos] - window.mean()) / window.std()
    )


def test_quality_spread_is_baa_minus_aaa_on_the_same_session():
    out = Credit().run(END)
    t = pd.Timestamp("2009-06-01")
    expected = series.load("DBAA", END)[t] - series.load("DAAA", END)[t]
    assert out.loc[t, "baa_aaa_level"] == pytest.approx(expected)


def test_features_start_populated_on_start_date():
    out = Credit().run(END)
    assert out.index[0] == Credit.start_date
    assert out.iloc[0].notna().all()


def test_holdout_is_locked():
    with pytest.raises(ValueError, match="holdout"):
        Credit().run("2018-01-02")


def test_exclude_backfill_blanks_the_window_and_what_depends_on_it():
    first, last = BACKFILL_SESSIONS
    base = Credit().run(END)
    robust = Credit(exclude_backfill=True).run(END)
    inside = (base.index >= first) & (base.index <= last)
    assert robust.loc[inside].isna().all().all()
    after = base.index[base.index > last]
    # Levels come back at once; the 20-session change once its lag clears the window.
    assert robust.loc[after[0], "baa10y_level"] == base.loc[after[0], "baa10y_level"]
    assert np.isnan(robust.loc[after[0], "baa10y_change_20"])
    assert robust.loc[after[20], "baa10y_change_20"] == base.loc[after[20], "baa10y_change_20"]
    # The z-score window still holds too few valid days through the end of 2017.
    assert robust.loc[after, "baa10y_z_252"].isna().all()
    # Nothing before the window changes.
    before = base.index < first
    pd.testing.assert_frame_equal(robust.loc[before], base.loc[before])


def test_exclude_backfill_has_no_lookahead():
    assert_no_lookahead(
        Credit(exclude_backfill=True), cutoffs=["2010-06-30", "2016-12-30", "2017-06-30"]
    )
