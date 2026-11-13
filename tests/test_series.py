import re
from pathlib import Path

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
        series.SeriesSpec("fake", "F1", pd.Timestamp("2010-01-04"), 1, 0, "test series"),
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


def test_refresh_refetches(fake_source):
    series.load("FAKE", "2010-03-31")
    series.load("FAKE", "2010-03-31", refresh=True)
    assert len(fake_source) == 2


def test_holdout_load_bypasses_the_cache(fake_source, tmp_path):
    out = series.load("FAKE", "2018-01-31", holdout=True)
    assert out.index[-1] == pd.Timestamp("2018-01-31")
    assert fake_source == [("F1", pd.Timestamp("2010-01-04"), pd.Timestamp("2018-01-31"))]
    assert not list(tmp_path.rglob("*.parquet"))


def test_every_series_has_a_fetcher_and_valid_rules():
    for name, spec in series.SERIES.items():
        assert spec.source in series.FETCHERS, name
        assert spec.lag >= 0 and spec.max_carry >= 0, name


def test_readme_documents_every_series():
    readme = (Path(__file__).resolve().parents[1] / "data" / "README.md").read_text()
    rows = {}
    for line in readme.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 7 and re.fullmatch(r"\d+", cells[4]):
            rows[cells[0]] = cells
    assert set(rows) == set(series.SERIES)
    for name, spec in series.SERIES.items():
        _, _, source_id, start, lag, carry, _ = rows[name]
        assert source_id.strip("`") == spec.source_id, name
        assert pd.Timestamp(start) == spec.start, name
        assert (int(lag), int(carry)) == (spec.lag, spec.max_carry), name
