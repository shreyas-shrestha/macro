import socket

import numpy as np
import pandas as pd
import pytest

from macro.data import cache, series
from macro.data.calendar import nyse_sessions
from macro.features.long_rv import LongRV
from macro.features.testing import assert_no_lookahead

SESSIONS = nyse_sessions("1985-01-02", "2017-12-29")
CLOSE = pd.Series(
    100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, len(SESSIONS)))), SESSIONS
)


@pytest.fixture(autouse=True)
def fake_spx(tmp_path, monkeypatch):
    def no_network(*args):
        raise AssertionError("tests must not hit the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    monkeypatch.setitem(series.FETCHERS, "yahoo", lambda sid, start, end: CLOSE.loc[start:end])


def test_long_rv_by_hand():
    out = LongRV().run("2017-12-29")
    returns = np.log(CLOSE).diff()
    t = pd.Timestamp("2008-09-15")
    for window in (126, 252):
        last = returns.loc[:t].iloc[-window:]
        assert out.loc[t, f"log_rv_{window}"] == pytest.approx(np.log(252 * (last**2).mean()))
    assert out.index[0] == pd.Timestamp("1990-01-02")
    assert out.notna().all().all()


def test_long_rv_no_lookahead():
    assert_no_lookahead(LongRV(), cutoffs=["1995-06-30", "2008-09-15", "2017-06-30"])
