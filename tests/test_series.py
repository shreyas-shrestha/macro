import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macro.config import DEV_END
from macro.data import cache, series
from macro.data.calendar import INDEX_NAME, nyse_sessions


@pytest.fixture
def fake_source(tmp_path, monkeypatch):
    """Register a fake source whose raw value on each business day is its day number."""
    calls = []

    def fetch(source_id, start, end):
        calls.append((source_id, start, end))
        days = pd.bdate_range(start, end)
        return pd.Series(range(len(days)), index=days, dtype=float)

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    monkeypatch.setitem(series.FETCHERS, "fake", fetch)
    monkeypatch.setitem(
        series.SERIES,
        "FAKE",
        series.SeriesSpec("fake", "F1", pd.Timestamp("2010-01-04"), 1, 0),
    )
    return calls


def test_load_returns_lagged_values_on_sessions(fake_source):
    out = series.load("FAKE", "2010-03-31")
    sessions = nyse_sessions("2010-01-04", "2010-03-31")
    assert out.index.equals(sessions)
    assert out.index.name == INDEX_NAME
    assert out.name == "FAKE"
    raw_day = pd.bdate_range("2010-01-04", "2010-03-31").get_loc(sessions[9])
    assert out.iloc[10] == raw_day


def test_load_refuses_holdout_dates(fake_source):
    with pytest.raises(ValueError, match="holdout"):
        series.load("FAKE", "2018-01-02")
    assert not fake_source


def test_load_fetches_once_through_dev_end_and_then_reads_the_cache(fake_source):
    series.load("FAKE", "2010-03-31")
    series.load("FAKE", "2012-06-29")
    assert fake_source == [("F1", pd.Timestamp("2010-01-04"), DEV_END)]


@pytest.mark.parametrize(
    "t", ["2010-06-30", "2010-07-03", "2010-04-02"], ids=["session", "weekend", "good_friday"]
)
def test_load_at_t_matches_a_later_load_up_to_t(fake_source, t):
    at_t = series.load("FAKE", t)  # cache miss
    full = series.load("FAKE", "2010-12-31")  # cache hit
    pd.testing.assert_series_equal(at_t, full.loc[:t])


def test_load_before_start_is_empty(fake_source):
    out = series.load("FAKE", "2009-12-31")
    assert out.empty and out.name == "FAKE" and out.index.name == INDEX_NAME
    assert not fake_source
    assert out.index.dtype == series.load("FAKE", "2010-03-31").index.dtype


@pytest.mark.parametrize(
    "raw, match",
    [
        (pd.Series([1.0], index=pd.DatetimeIndex(["2010-01-05"], tz="UTC")), "tz-naive"),
        (pd.Series([1.0], index=pd.to_datetime(["2010-01-05 16:15"])), "time of day"),
        (pd.Series([1], index=pd.to_datetime(["2010-01-05"])), "floats"),
        (pd.Series([1.0], index=pd.to_datetime(["2018-01-02"])), "outside"),
        (pd.Series([np.nan], index=pd.to_datetime(["2010-01-05"])), "no data"),
    ],
)
def test_fetcher_contract_violations_raise_before_caching(
    fake_source, tmp_path, monkeypatch, raw, match
):
    monkeypatch.setitem(series.FETCHERS, "fake", lambda *args: raw)
    with pytest.raises(ValueError, match=match):
        series.load("FAKE", "2010-03-31")
    assert not list(tmp_path.rglob("*"))


def test_bad_cache_file_is_rejected_on_read(fake_source, tmp_path):
    path = tmp_path / "fake" / "F1.parquet"
    path.parent.mkdir()
    pd.Series([1.0], index=pd.to_datetime(["2018-06-01"])).rename("value").to_frame().to_parquet(
        path
    )
    with pytest.raises(ValueError, match="outside"):
        series.load("FAKE", "2010-03-31")


def test_refresh_refetches(fake_source):
    series.load("FAKE", "2010-03-31")
    series.load("FAKE", "2010-03-31", refresh=True)
    assert len(fake_source) == 2


def test_holdout_load_bypasses_the_cache(fake_source, tmp_path):
    out = series.load("FAKE", "2018-01-31", holdout=True)
    assert out.index[-1] == pd.Timestamp("2018-01-31")
    assert fake_source == [("F1", pd.Timestamp("2010-01-04"), pd.Timestamp("2018-01-31"))]
    assert not list(tmp_path.rglob("*.parquet"))


def test_every_series_has_a_fetcher():
    for name, spec in series.SERIES.items():
        assert spec.source in series.FETCHERS, name


def test_readme_documents_every_series():
    readme = (Path(__file__).resolve().parents[1] / "data" / "README.md").read_text()
    rows = {}
    for line in readme.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 8 and re.fullmatch(r"\d+", cells[4]):
            rows[cells[0]] = cells
    assert set(rows) == set(series.SERIES)
    for name, spec in series.SERIES.items():
        _, _, source_id, start, lag, carry, _, _ = rows[name]
        assert source_id.strip("`") == spec.source_id, name
        assert pd.Timestamp(start) == spec.start, name
        assert (int(lag), int(carry)) == (spec.lag, spec.max_carry), name
