import pandas as pd
import pytest
import yfinance as yf

from macro.data import yahoo


def test_fetch_requests_through_end_and_trims_extra_rows(monkeypatch):
    calls = []

    def download(ticker, **kwargs):
        calls.append(kwargs)
        index = pd.to_datetime(["2017-12-28", "2017-12-29", "2018-01-02"])
        return pd.DataFrame({"Close": [1, 2, 3]}, index=index)

    monkeypatch.setattr(yf, "download", download)
    out = yahoo.fetch("^VIX", pd.Timestamp("2017-12-01"), pd.Timestamp("2017-12-31"))
    assert calls[0]["end"] == pd.Timestamp("2018-01-01")
    assert list(out) == [1.0, 2.0]


def test_empty_response_raises(monkeypatch):
    monkeypatch.setattr(yf, "download", lambda ticker, **kwargs: pd.DataFrame())
    with pytest.raises(RuntimeError, match="no data"):
        yahoo.fetch("^BAD", pd.Timestamp("2017-01-03"), pd.Timestamp("2017-12-29"))
