import io
import json
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import pytest

from macro.data import alfred, bcrp, fred
from macro.data.calendar import nyse_sessions
from macro.data.pit import align


@pytest.fixture
def respond(monkeypatch):
    """Serve ``payload`` as JSON to urlopen and record the requested URLs."""
    urls = []

    def install(payload):
        def urlopen(url, timeout):
            urls.append(url)
            return io.BytesIO(json.dumps(payload).encode())

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


def test_fred_requires_an_api_key(monkeypatch):
    monkeypatch.setattr(fred, "load_env", lambda: False)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FRED_API_KEY"):
        fred.fetch("BAA10Y", pd.Timestamp("2010-01-04"), pd.Timestamp("2010-12-31"))


def test_fred_fetch_trims_to_the_range(monkeypatch):
    class FakeFred:
        def __init__(self, api_key):
            pass

        def get_series(self, series_id, observation_start, observation_end):
            return pd.Series([1, 2], index=pd.to_datetime(["2010-12-31", "2011-01-03"]))

    monkeypatch.setenv("FRED_API_KEY", "test")
    monkeypatch.setattr(fred, "Fred", FakeFred)
    out = fred.fetch("BAA10Y", pd.Timestamp("2010-01-04"), pd.Timestamp("2010-12-31"))
    assert list(out) == [1.0]
