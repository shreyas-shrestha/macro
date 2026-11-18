import io
import json
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import pytest

from macro.data import alfred, bcrp, claims, fred
from macro.data.calendar import nyse_sessions
from macro.data.pit import align


@pytest.fixture
def respond(monkeypatch):
    """Serve ``payload`` to urlopen (a dict as JSON) and record requested URLs."""
    urls = []

    def install(payload):
        def urlopen(request, timeout):
            urls.append(getattr(request, "full_url", request))
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            return io.BytesIO(body)

        monkeypatch.setattr(urllib.request, "urlopen", urlopen)
        return urls

    return install


def test_bcrp_parse_handles_spanish_months_and_missing_values():
    periods = [
        {"name": "29.Dic.99", "values": ["466"]},
        {"name": "01.Set.17", "values": ["n.d."]},
    ]
    out = bcrp.parse(periods)
    assert list(out.index) == [pd.Timestamp("1999-12-29"), pd.Timestamp("2017-09-01")]
    assert out.iloc[0] == 466.0 and np.isnan(out.iloc[1])


def test_bcrp_fetch_requests_the_range_and_trims(respond):
    urls = respond(
        {
            "periods": [
                {"name": "29.Dic.17", "values": ["1"]},
                {"name": "02.Ene.18", "values": ["2"]},
            ]
        }
    )
    out = bcrp.fetch("PD04708XD", pd.Timestamp("2017-12-01"), pd.Timestamp("2017-12-31"))
    assert urls[0].endswith("/PD04708XD/json/2017-12-01/2017-12-31")
    assert list(out) == [1.0]


def test_bcrp_non_json_page_raises(respond):
    respond(b"<html>challenge</html>")
    with pytest.raises(RuntimeError, match="PD04708XD"):
        bcrp.fetch("PD04708XD", pd.Timestamp("2017-12-01"), pd.Timestamp("2017-12-31"))


def test_alfred_fetch_asks_for_first_releases_published_by_end(respond, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test")
    rows = [
        {"date": "2009-12-25", "realtime_start": "2009-12-30", "value": "1.0"},
        {"date": "2017-12-29", "realtime_start": "2017-12-31", "value": "2.0"},
    ]
    urls = respond({"observations": rows})
    out = alfred.fetch("NFCI", pd.Timestamp("2010-01-01"), pd.Timestamp("2017-12-31"))
    query = urllib.parse.parse_qs(urllib.parse.urlparse(urls[0]).query)
    assert query["output_type"] == ["4"]
    assert query["realtime_end"] == ["2017-12-31"]
    # Stamped by publication date, so the 2009 release falls before start.
    assert list(out.index) == [pd.Timestamp("2017-12-31")]


def test_alfred_back_history_appears_once_on_its_first_publication_date():
    # ALFRED first tracked this series on 2011-05-25 with ten years of history; later
    # weeks were published one at a time.
    back = [
        {"date": d.strftime("%Y-%m-%d"), "realtime_start": "2011-05-25", "value": str(i)}
        for i, d in enumerate(pd.date_range("2001-01-05", "2011-05-20", freq="W-FRI"))
    ]
    new = [{"date": "2011-05-27", "realtime_start": "2011-06-01", "value": "-1"}]
    raw = alfred.parse(back[::-1] + new)  # arrival order must not matter
    sessions = nyse_sessions("2011-05-02", "2011-06-10")
    out = align(raw, sessions, lag=1, max_carry=10)
    assert out.loc[:"2011-05-25"].isna().all()
    assert out[pd.Timestamp("2011-05-26")] == len(back) - 1
    assert out[pd.Timestamp("2011-06-02")] == -1.0


def test_alfred_parse_treats_dot_as_missing():
    out = alfred.parse([{"date": "2011-05-20", "realtime_start": "2011-05-25", "value": "."}])
    assert out.dtype == float and np.isnan(out.iloc[0])


def test_alfred_parse_returns_floats_for_whole_numbers():
    out = alfred.parse([{"date": "2011-05-21", "realtime_start": "2011-05-26", "value": "424000"}])
    assert out.dtype == float


def test_fred_requires_an_api_key(monkeypatch):
    monkeypatch.setattr(fred, "load_env", lambda: False)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FRED_API_KEY"):
        fred.fetch("BAA10Y", pd.Timestamp("2010-01-04"), pd.Timestamp("2010-12-31"))


def test_fred_values_are_stamped_by_fed_posting_day(monkeypatch):
    observed = pd.to_datetime(["2016-11-10", "2017-10-06", "2017-12-29"])

    class FakeFred:
        def __init__(self, api_key):
            pass

        def get_series(self, series_id, observation_start, observation_end):
            return pd.Series([1, 2, 3], index=observed)

    monkeypatch.setenv("FRED_API_KEY", "test")
    monkeypatch.setattr(fred, "Fred", FakeFred)
    out = fred.fetch("DGS10", pd.Timestamp("2016-01-04"), pd.Timestamp("2017-12-31"))
    # Veterans Day (2016-11-11) and Columbus Day (2017-10-09) are NYSE sessions on which
    # FRED does not post; the 12-29 value is posted in 2018 and dropped.
    assert list(out.index) == [pd.Timestamp("2016-11-14"), pd.Timestamp("2017-10-10")]
    aligned = align(out.loc["2017"], nyse_sessions("2017-10-02", "2017-10-13"), lag=1, max_carry=0)
    assert aligned.first_valid_index() == pd.Timestamp("2017-10-11")


@pytest.fixture
def claims_sources(monkeypatch):
    """Serve a vintage and ALFRED first releases to ``claims.fetch``; record requests."""
    requests = []

    def install(vintage, first):
        class FakeFred:
            def __init__(self, api_key):
                pass

            def get_series(self, series_id, observation_start, observation_end, **kwargs):
                requests.append(kwargs)
                return vintage

        monkeypatch.setenv("FRED_API_KEY", "test")
        monkeypatch.setattr(claims, "Fred", FakeFred)
        monkeypatch.setattr(alfred, "first_releases", lambda series_id, end: alfred.table(first))
        return requests

    return install


def test_claims_use_first_releases_where_alfred_has_them(claims_sources):
    weeks = pd.to_datetime(["2009-05-23", "2009-05-30", "2017-11-18", "2017-12-23", "2017-12-30"])
    vintage = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=weeks)
    first = [
        {"date": "2009-05-30", "realtime_start": "2009-06-04", "value": "20"},
        # Thanksgiving: released Wednesday 11-22.
        {"date": "2017-11-18", "realtime_start": "2017-11-22", "value": "30"},
        {"date": "2017-12-23", "realtime_start": "2017-12-28", "value": "40"},
    ]
    requests = claims_sources(vintage, first)
    out = claims.fetch("ICNSA", pd.Timestamp("2009-01-01"), pd.Timestamp("2017-12-31"))
    assert requests == [{"realtime_start": "2017-12-31", "realtime_end": "2017-12-31"}]
    # 2009-05-23 has no first release (ALFRED omits its first vintage) and keeps the
    # vintage value; the week ending 2017-12-30 is released in 2018 and dropped.
    expected_index = pd.to_datetime(["2009-05-28", "2009-06-04", "2017-11-23", "2017-12-28"])
    pd.testing.assert_series_equal(out, pd.Series([1.0, 20.0, 30.0, 40.0], index=expected_index))
    # Stamped Thursday 11-23 (an NYSE holiday), so first used Monday 11-27.
    aligned = align(out.loc["2017-11"], nyse_sessions("2017-11-20", "2017-12-01"), 1, 10)
    assert aligned.first_valid_index() == pd.Timestamp("2017-11-27")


def test_claims_refuse_a_release_after_the_stamp(claims_sources):
    vintage = pd.Series([1.0], index=pd.to_datetime(["2013-10-05"]))
    claims_sources(vintage, [{"date": "2013-10-05", "realtime_start": "2013-10-11", "value": "2"}])
    with pytest.raises(ValueError, match="after the stamp"):
        claims.fetch("ICNSA", pd.Timestamp("2013-01-01"), pd.Timestamp("2017-12-31"))


def test_claims_refuse_weeks_not_ending_saturday(claims_sources):
    claims_sources(pd.Series([1.0], index=pd.to_datetime(["2017-11-17"])), [])
    with pytest.raises(ValueError, match="Saturday"):
        claims.fetch("ICNSA", pd.Timestamp("2017-01-01"), pd.Timestamp("2017-12-31"))
