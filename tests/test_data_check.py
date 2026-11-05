import runpy
from pathlib import Path
from unittest.mock import Mock

import fredapi
import pandas as pd
import yfinance as yf

from macro import config


def test_data_check_requests_only_development_sample(monkeypatch):
    prices = pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2017-12-29"]))
    download = Mock(return_value=prices)
    fred = Mock()
    fred.get_series.return_value = prices["Close"]
    monkeypatch.setattr(yf, "download", download)
    monkeypatch.setattr(fredapi, "Fred", Mock(return_value=fred))
    monkeypatch.setattr(config, "load_env", lambda: False)
    monkeypatch.setenv("FRED_API_KEY", "test")

    runpy.run_path(Path(__file__).resolve().parents[1] / "data" / "data_check.py")

    assert download.call_count == 2
    download.assert_any_call("^GSPC", start=config.DEV_START, end=config.HOLDOUT_START)
    download.assert_any_call("^VIX", start=config.DEV_START, end=config.HOLDOUT_START)
    fred.get_series.assert_called_once_with(
        "BAA10Y", observation_start=config.DEV_START, observation_end=config.DEV_END
    )
